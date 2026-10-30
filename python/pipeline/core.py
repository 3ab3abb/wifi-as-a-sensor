"""The whole pipeline (steps 4-9) in one place, offline and live.

All tunable numbers live in `PipelineConfig`, so every experiment ("what if cutoff = 10 Hz?")
is one changed field, and the calibration file records exactly which settings it was made with.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import filters
from .detect import Decisions, Detector, calibrate_threshold, decide
from .hampel import StreamingHampel, hampel
from .measure import activity_series, window_activity
from .parse import LinkData
from .resample import StreamingResampler, normalize_packets, resample_uniform


@dataclass
class PipelineConfig:
    fs: float = 100.0                 # uniform sampling rate after resampling (Hz)
    max_gap_s: float = 0.05           # longer packet gaps make a window untrusted
    normalize: bool = True            # per-packet AGC normalization
    hampel_window: int = 11           # samples (odd)
    hampel_sigmas: float = 3.0
    lp_order: int = 4                 # Butterworth order
    lp_cutoff_hz: float = 20.0
    zero_phase: bool = True           # offline: filtfilt; False = causal like live
    n_components: int = 3             # PCA components kept
    drop_first_pc: bool = False
    pca_window_s: float = 10.0        # trailing history used to fit PCA
    activity_window_s: float = 1.0    # moving-variance window
    hop_s: float = 0.5                # one decision every hop
    threshold_method: str = "sigma"   # "sigma" (mean + k*std) or "percentile"
    threshold_k: float = 3.0
    threshold_percentile: float = 99.0
    n_on: int = 2                     # hysteresis (windows)
    n_off: int = 4
    hold_s: float = 30.0              # presence hold

    def sos(self) -> np.ndarray:
        return filters.butter_lowpass(self.lp_order, self.lp_cutoff_hz, self.fs)


@dataclass
class Stages:
    """Every intermediate signal, kept for checkpoint plots."""
    t_packets: np.ndarray
    amp_packets: np.ndarray           # step 4: raw amplitude per packet
    t: np.ndarray                     # uniform grid
    amp: np.ndarray                   # step 4: normalized + resampled
    gap_mask: np.ndarray
    cleaned: np.ndarray               # step 5
    outliers: np.ndarray
    smoothed: np.ndarray              # step 6
    t_act: np.ndarray                 # step 8
    activity: np.ndarray
    valid: np.ndarray                 # window free of packet gaps
    decisions: Decisions | None = None  # step 9 (needs a threshold)


def run_offline(link: LinkData, cfg: PipelineConfig = PipelineConfig(),
                threshold: float | None = None) -> Stages:
    amp_p = normalize_packets(link.amp) if cfg.normalize else link.amp
    t, amp, gaps = resample_uniform(link.t, amp_p, cfg.fs, max_gap_s=cfg.max_gap_s)
    cleaned, outliers = hampel(amp, cfg.hampel_window, cfg.hampel_sigmas)
    sos = cfg.sos()
    smoothed = filters.apply_zero_phase(sos, cleaned) if cfg.zero_phase else filters.apply_causal(sos, cleaned)
    t_act, act, valid = activity_series(smoothed, cfg.fs, cfg.activity_window_s, cfg.hop_s,
                                        cfg.pca_window_s, cfg.n_components, cfg.drop_first_pc, gaps)
    t_act = t_act + t[0]
    st = Stages(link.t, link.amp, t, amp, gaps, cleaned, outliers, smoothed, t_act, act, valid)
    if threshold is not None:
        st.decisions = decide(act, threshold, cfg.hop_s, cfg.n_on, cfg.n_off, cfg.hold_s, valid)
    return st


def threshold_from_activity(activity: np.ndarray, cfg: PipelineConfig) -> float:
    """activity: values from VALID windows of an empty-room recording."""
    return calibrate_threshold(activity, cfg.threshold_method, cfg.threshold_k, cfg.threshold_percentile)


# --- calibration file -------------------------------------------------------------------

def save_calibration(path: str | Path, threshold: float, cfg: PipelineConfig, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"threshold": threshold, "config": asdict(cfg), **extra}, indent=2))


def load_calibration(path: str | Path) -> tuple[float, PipelineConfig]:
    d = json.loads(Path(path).read_text())
    return d["threshold"], PipelineConfig(**d["config"])


# --- live -------------------------------------------------------------------------------

@dataclass
class LiveOutput:
    t: float
    activity: float
    valid: bool
    motion: bool
    present: bool


class StreamingPipeline:
    """Same steps as run_offline, but packet by packet with only past data:
    streaming resampler -> streaming Hampel (50 ms delay) -> causal Butterworth ->
    ring buffer -> activity every hop -> Detector. Threshold may be set later
    (after a live calibration period)."""

    def __init__(self, cfg: PipelineConfig, threshold: float | None = None):
        self.cfg = cfg
        self.resampler = StreamingResampler(cfg.fs, cfg.max_gap_s)
        self.hampel = StreamingHampel(cfg.hampel_window, cfg.hampel_sigmas)
        self.filter = filters.StreamingFilter(cfg.sos())
        self.pca_n = int(round(cfg.pca_window_s * cfg.fs))
        self.win_n = int(round(cfg.activity_window_s * cfg.fs))
        self.hop_n = int(round(cfg.hop_s * cfg.fs))
        self.buffer: np.ndarray | None = None
        self.gap_flags: list[bool] = []     # aligned with Hampel input; trimmed as output appears
        self.recent_gaps: list[bool] = []   # aligned with buffer rows (last win_n)
        self._hampel_started = False
        self.t_last = 0.0
        self._since_hop = 0
        self.detector: Detector | None = None
        if threshold is not None:
            self.set_threshold(threshold)

    def set_threshold(self, threshold: float):
        c = self.cfg
        self.detector = Detector(threshold, c.hop_s, c.n_on, c.n_off, c.hold_s)

    def push_packet(self, t: float, amp: np.ndarray) -> list[LiveOutput]:
        if self.cfg.normalize:
            amp = normalize_packets(amp[None, :])[0]
        samples = self.resampler.push(t, amp)
        if not samples:
            return []
        self.t_last = samples[-1][0]
        self.gap_flags += [g for _, _, g in samples]
        y = self.filter.push(self.hampel.push(np.array([s for _, s, _ in samples])))
        if len(y) and not self._hampel_started:
            # the Hampel output starts half a window after its input
            self.gap_flags = self.gap_flags[self.cfg.hampel_window // 2:]
            self._hampel_started = True
        out = []
        for row in y:
            self.buffer = row[None] if self.buffer is None else np.vstack([self.buffer[-(self.pca_n - 1):], row])
            self.recent_gaps = (self.recent_gaps + [self.gap_flags.pop(0)])[-self.win_n:]
            self._since_hop += 1
            if self._since_hop >= self.hop_n and self.buffer.shape[0] >= self.pca_n:
                self._since_hop = 0
                a = window_activity(self.buffer, self.win_n, self.cfg.n_components, self.cfg.drop_first_pc)
                valid = not any(self.recent_gaps)
                motion, present = self.detector.update(a, valid) if self.detector else (False, False)
                out.append(LiveOutput(self.t_last, a, valid, motion, present))
        return out
