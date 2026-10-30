"""Live motion detection demo.

Reads CSI from the ESP32 (or replays a file), runs the causal streaming pipeline and
shows the activity level, threshold and decision in a window.

Calibration, either:
  --calib ../data/processed/calibration.json   threshold measured offline (analysis.evaluate --save-calib)
  --calibrate 30                               leave the room for the first 30 s; threshold measured live

Usage (from python/):
    python -m live.live_detect --calibrate 30                       # ESP32 on USB
    python -m live.live_detect --replay ../data/synthetic/sim_mixed_1.csv \
        --calib ../data/processed/calibration_synthetic.json        # no hardware needed
    add --headless to print decisions instead of plotting
"""
from __future__ import annotations

import argparse
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np

from capture.sources import ReplaySource, SerialSource
from pipeline.core import PipelineConfig, StreamingPipeline, load_calibration, threshold_from_activity
from pipeline.parse import TIMESTAMP_WRAP_US, amplitude, parse_line, select_subcarriers


class LiveDetector:
    """Glue: lines -> packets -> one StreamingPipeline per link (TX MAC address)."""

    def __init__(self, cfg: PipelineConfig, threshold: float | None, calibrate_s: float):
        self.cfg, self.threshold, self.calibrate_s = cfg, threshold, calibrate_s
        self.pipes: dict[str, StreamingPipeline] = {}
        self.clock: dict[str, list[int]] = {}       # mac -> [first_ts, last_ts, wrap_offset]
        self.calib_values: list[float] = []
        self.history = deque(maxlen=int(120 / cfg.hop_s))   # (t, activity, valid, motion, present)

    def _time(self, mac: str, ts: int) -> float:
        c = self.clock.setdefault(mac, [ts, ts, 0])
        if ts < c[1] - TIMESTAMP_WRAP_US // 2:
            c[2] += TIMESTAMP_WRAP_US
        c[1] = ts
        return (ts + c[2] - c[0]) / 1e6

    def feed(self, line: str):
        pkt = parse_line(line)
        if pkt is None:
            return
        pipe = self.pipes.get(pkt.mac)
        if pipe is None:
            pipe = self.pipes[pkt.mac] = StreamingPipeline(self.cfg, self.threshold)
            print(f"new link: {pkt.mac}")
        if pkt.mac != self.main_mac:
            pipe.push_packet(self._time(pkt.mac, pkt.timestamp_us), amplitude(select_subcarriers(pkt)))
            return  # multi-link fusion is a stretch goal; only the main link is shown
        for o in pipe.push_packet(self._time(pkt.mac, pkt.timestamp_us), amplitude(select_subcarriers(pkt))):
            if self.threshold is None:
                if o.valid:
                    self.calib_values.append(o.activity)
                if len(self.calib_values) * self.cfg.hop_s >= self.calibrate_s:
                    self.threshold = threshold_from_activity(np.array(self.calib_values), self.cfg)
                    for p in self.pipes.values():
                        p.set_threshold(self.threshold)
                    print(f"calibrated: threshold = {self.threshold:.4g}")
            self.history.append((o.t, o.activity, o.valid, o.motion, o.present))

    @property
    def main_mac(self) -> str | None:
        return next(iter(self.pipes), None)   # first link seen (router mode: the only one)


def reader_thread(source, q: queue.Queue):
    for _, line in source:
        q.put(line)
    q.put(None)


def run_headless(det: LiveDetector, q: queue.Queue):
    shown = 0
    while (line := q.get()) is not None:
        det.feed(line)
        while shown < len(det.history):
            t, a, valid, motion, present = det.history[shown]
            shown += 1
            state = "calibrating" if det.threshold is None else ("MOTION" if motion else "quiet") + (" / present" if present else "")
            print(f"t={t:7.1f}s activity={a:.4g}{'' if valid else ' (gap)'}  {state}")


def run_gui(det: LiveDetector, q: queue.Queue, window_s: float = 60):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 4.5))
    line, = ax.semilogy([], [], color="#2a78d6", lw=2)
    thr_line = ax.axhline(np.nan, color="#0b0b0b", ls="--", lw=1)
    status = fig.text(0.5, 0.93, "waiting for CSI...", ha="center", fontsize=22, fontweight="bold")
    ax.set(xlabel="time (s)", ylabel="activity (log)")
    fig.subplots_adjust(top=0.85)

    def update(_):
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            det.feed(item)
        if not det.history:
            return line,
        h = np.array([(t, a if v else np.nan, m, p) for t, a, v, m, p in det.history])
        t, a = h[:, 0], h[:, 1]
        line.set_data(t, a)
        ax.set_xlim(max(t[-1] - window_s, t[0]), t[-1] + 1)
        finite = a[np.isfinite(a)]
        if finite.size:
            lo, hi = finite.min(), finite.max()
            if det.threshold is not None:
                lo, hi = min(lo, det.threshold), max(hi, det.threshold)
                thr_line.set_ydata([det.threshold] * 2)
            ax.set_ylim(lo / 2, hi * 2)
        if det.threshold is None:
            left = det.calibrate_s - len(det.calib_values) * det.cfg.hop_s
            status.set_text(f"CALIBRATING — keep the room empty ({left:.0f} s)")
            status.set_color("#52514e")
        elif h[-1, 2]:
            status.set_text("MOTION"); status.set_color("#eb6834")
        elif h[-1, 3]:
            status.set_text("PRESENT (no motion now)"); status.set_color("#2a78d6")
        else:
            status.set_text("EMPTY"); status.set_color("#52514e")
        return line, thr_line, status

    _anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib", type=Path)
    g.add_argument("--calibrate", type=float, metavar="SECONDS")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if args.calib:
        threshold, cfg = load_calibration(args.calib)
        if cfg.zero_phase:
            print("warning: this calibration was made with the zero-phase filter; for live use, "
                  "recalibrate with `analysis.evaluate --causal --save-calib ...`")
    else:
        threshold, cfg = None, PipelineConfig()
    cfg.zero_phase = False   # live = causal, always
    det = LiveDetector(cfg, threshold, args.calibrate or 0)
    src = ReplaySource(args.replay, args.speed) if args.replay else SerialSource(args.port)
    q: queue.Queue = queue.Queue()
    threading.Thread(target=reader_thread, args=(src, q), daemon=True).start()
    (run_headless if args.headless else run_gui)(det, q)


if __name__ == "__main__":
    main()
