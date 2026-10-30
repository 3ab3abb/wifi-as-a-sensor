"""Checkpoint plots for steps 1 and 4-9 — each one becomes a slide.

Usage (from python/):
    python -m analysis.checkpoints --empty ../data/raw/empty.csv --walking ../data/raw/walking.csv \
        --mixed ../data/raw/mixed.csv --calib ../data/processed/calibration.json --out ../figures

Recordings marked "synthetic" in their .json get a SYNTHETIC DATA watermark: those plots
test the code, they are not results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pipeline import filters
from pipeline.combine import pca
from pipeline.core import PipelineConfig, Stages, load_calibration, run_offline
from pipeline.parse import DATA_SUBCARRIERS, load_recording
from pipeline.resample import timing_stats
from analysis.evaluate import label_segments

# Colors (validated categorical palette, light mode) and text inks.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.5,
    "legend.frameon": False,
})


def _plot_activity(ax, st: Stages, color, label):
    """Trusted windows as a line; windows containing a packet gap as faint dots."""
    ax.semilogy(st.t_act, np.where(st.valid, st.activity, np.nan), color=color, label=label)
    ax.semilogy(st.t_act[~st.valid], st.activity[~st.valid], ".", color=color, alpha=0.35, ms=4)


def _finish(fig, path: Path, synthetic: bool):
    if synthetic:
        fig.text(0.99, 0.01, "SYNTHETIC DATA — pipeline test, not a result", ha="right",
                 va="bottom", color=INK2, fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def step1_timing(t, out, synthetic):
    s = timing_stats(t)
    dt = np.diff(t) * 1e3
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(np.clip(dt, 0, 50), bins=100, color=BLUE)
    ax.set(xlabel="interval between packets (ms, clipped at 50)", ylabel="packets",
           title="Step 1 — packet timing = our sampling")
    ax.text(0.98, 0.95, f"rate {s.mean_rate_hz:.1f} Hz\njitter (std) {s.interval_std_ms:.1f} ms\n"
            f"99th pct {s.interval_p99_ms:.1f} ms\nlongest gap {s.longest_gap_ms:.0f} ms",
            transform=ax.transAxes, ha="right", va="top", color=INK)
    _finish(fig, out / "step1_timing.png", synthetic)


def step4_heatmaps(st_e: Stages, st_w: Stages, out, synthetic, seconds=30):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    lo, hi = np.percentile(np.r_[st_e.amp.ravel(), st_w.amp.ravel()], [1, 99])
    for ax, st, name in [(axes[0], st_e, "empty room"), (axes[1], st_w, "person walking")]:
        n = int(seconds * 100)
        im = ax.imshow(st.amp[:n].T, aspect="auto", origin="lower", cmap="Blues", vmin=lo, vmax=hi,
                       extent=[0, n / 100, DATA_SUBCARRIERS[0], DATA_SUBCARRIERS[-1]])
        ax.set(title=name, xlabel="time (s)")
        ax.grid(False)
    axes[0].set_ylabel("subcarrier index k")
    fig.colorbar(im, ax=axes, label="normalized amplitude |H(k)|", shrink=0.9)
    fig.suptitle("Step 4 — CSI amplitude: time x subcarrier", fontweight="bold", y=1.04)
    # no tight_layout here: it fights with the shared colorbar
    if synthetic:
        fig.text(0.99, 0.0, "SYNTHETIC DATA — pipeline test, not a result", ha="right",
                 color=INK2, fontsize=8, style="italic")
    fig.savefig(out / "step4_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "step4_heatmap.png")


def step5_hampel(st: Stages, out, synthetic, seconds=10):
    # show the subcarrier with the most outliers, over the busiest 10 s
    k = int(st.outliers.sum(axis=0).argmax())
    n = int(seconds * 100)
    counts = np.convolve(st.outliers[:, k], np.ones(n), "valid")
    i0 = int(counts.argmax())
    sl = slice(i0, i0 + n)
    t = st.t[sl] - st.t[i0]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, st.amp[sl, k], color=INK2, lw=1, label="before (resampled)")
    ax.plot(t, st.cleaned[sl, k], color=BLUE, label="after Hampel")
    o = st.outliers[sl, k]
    ax.scatter(t[o], st.amp[sl, k][o], s=30, color=ORANGE, zorder=3, label="replaced outliers",
               edgecolors="white", linewidths=1)
    ax.set(xlabel="time (s)", ylabel=f"amplitude, subcarrier k={DATA_SUBCARRIERS[k]}",
           title="Step 5 — Hampel filter removes spikes, keeps the rest")
    ax.legend(loc="upper right")
    _finish(fig, out / "step5_hampel.png", synthetic)


def step6_response(cfg: PipelineConfig, out, synthetic):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for order, c in [(2, AQUA), (4, BLUE), (8, VIOLET)]:
        f, mag, gd = filters.frequency_response(filters.butter_lowpass(order, cfg.lp_cutoff_hz, cfg.fs), cfg.fs)
        ax1.plot(f, mag, color=c, label=f"order {order}")
        ax2.plot(f, gd / cfg.fs * 1e3, color=c, label=f"order {order}")
    ax1.axvline(cfg.lp_cutoff_hz, color=INK2, ls="--", lw=1)
    ax1.axhline(-3, color=INK2, ls=":", lw=1)
    ax1.text(cfg.lp_cutoff_hz + 0.5, -55, f"fc = {cfg.lp_cutoff_hz:g} Hz", color=INK2)
    ax1.set(xlim=(0, cfg.fs / 2), ylim=(-60, 5), xlabel="frequency (Hz)", ylabel="|H(f)| (dB)",
            title="Butterworth magnitude response")
    ax2.set(xlim=(0, cfg.fs / 2), xlabel="frequency (Hz)", ylabel="group delay (ms)",
            title="Delay of the causal filter")
    ax1.legend(); ax2.legend()
    fig.suptitle("Step 6 — Butterworth low-pass design (fs = 100 Hz)", fontweight="bold")
    _finish(fig, out / "step6_response.png", synthetic)


def step6_spectra(st_e: Stages, st_w: Stages, cfg, out, synthetic):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for st, name, c in [(st_e, "empty", INK2), (st_w, "walking", ORANGE)]:
        x = st.cleaned - st.cleaned.mean(axis=0)
        f, p = signal.welch(x, fs=cfg.fs, nperseg=256, axis=0)
        ax.semilogy(f, p.mean(axis=1), color=c, lw=1, ls="--", label=f"{name}, before low-pass")
        y = st.smoothed - st.smoothed.mean(axis=0)
        f, p = signal.welch(y, fs=cfg.fs, nperseg=256, axis=0)
        ax.semilogy(f, p.mean(axis=1), color=c, label=f"{name}, after low-pass")
    ax.axvline(cfg.lp_cutoff_hz, color=INK2, ls=":", lw=1)
    top = ax.get_ylim()[1]
    ax.set_ylim(top * 1e-6, top)
    ax.set(xlabel="frequency (Hz)", ylabel="PSD (avg over subcarriers)",
           title="Step 6 — spectra: motion lives at low frequencies; low-pass removes the rest")
    ax.legend(fontsize=8)
    _finish(fig, out / "step6_spectra.png", synthetic)


def step6_zero_phase(st_w: Stages, cfg, out, synthetic, seconds=4):
    k = int(st_w.cleaned.var(axis=0).argmax())
    n = int(seconds * cfg.fs)
    i0 = st_w.cleaned.shape[0] // 2
    x = st_w.cleaned[:, k]
    sos = cfg.sos()
    zp = filters.apply_zero_phase(sos, x[:, None])[:, 0]
    ca = filters.apply_causal(sos, x[:, None])[:, 0]
    t = np.arange(n) / cfg.fs
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, x[i0:i0 + n], color=GRID, lw=2.5, label="input (after Hampel)")
    ax.plot(t, zp[i0:i0 + n], color=BLUE, label="zero-phase (filtfilt, offline)")
    ax.plot(t, ca[i0:i0 + n], color=ORANGE, label="causal (sosfilt, live)")
    ax.set(xlabel="time (s)", ylabel=f"amplitude, k={DATA_SUBCARRIERS[k]}",
           title="Step 6 — zero-phase vs causal: same shape, causal is delayed")
    ax.legend(fontsize=8)
    _finish(fig, out / "step6_zero_phase_vs_causal.png", synthetic)


def step7_pca(st_e: Stages, st_w: Stages, cfg, out, synthetic, seconds=20):
    pe, pw = pca(st_e.smoothed, 10), pca(st_w.smoothed, 10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    idx = np.arange(1, 11)
    ax1.bar(idx - 0.2, pe.explained_ratio * 100, 0.38, color=INK2, label="empty")
    ax1.bar(idx + 0.2, pw.explained_ratio * 100, 0.38, color=ORANGE, label="walking")
    ax1.set(xlabel="principal component", ylabel="explained variance (%)", xticks=idx,
            title="Explained variance")
    ax1.legend()
    n = int(seconds * cfg.fs)
    t = np.arange(n) / cfg.fs
    for i, c in zip(range(3), [BLUE, ORANGE, AQUA]):
        ax2.plot(t, pw.scores[:n, i], color=c, lw=1, label=f"PC{i + 1}")
    ax2.set(xlabel="time (s)", ylabel="component value", title="Top 3 components (walking)")
    ax2.legend(ncol=3, loc="upper right")
    fig.suptitle("Step 7 — background removal + PCA combine 52 subcarriers into a few signals",
                 fontweight="bold")
    _finish(fig, out / "step7_pca.png", synthetic)


def step8_activity(curves, threshold, out, synthetic):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for (name, st), c in zip(curves, [INK2, ORANGE, BLUE]):
        _plot_activity(ax, st, c, name)
    if threshold is not None:
        ax.axhline(threshold, color=INK, ls="--", lw=1)
        ax.text(ax.get_xlim()[0], threshold * 1.15, " threshold", color=INK, fontsize=8)
    ax.set(xlabel="time (s)", ylabel="activity (moving variance, log)",
           title="Step 8 — activity level: 1 s moving variance of top PCA components")
    ax.legend(title="dots = window with a\npacket gap (ignored)", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    _finish(fig, out / "step8_activity.png", synthetic)


def step9_decision(st: Stages, segments, threshold, out, synthetic):
    d = st.decisions
    fig, (ax, axt) = plt.subplots(2, 1, figsize=(9, 4), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    _plot_activity(ax, st, BLUE, "activity (dots: packet gap, ignored)")
    ax.axhline(threshold, color=INK, ls="--", lw=1, label="threshold (empty-room mean + 3σ)")
    ylo, yhi = ax.get_ylim()
    ax.fill_between(st.t_act, ylo, yhi, where=d.motion, color=ORANGE, alpha=0.18, step="mid",
                    label="decision: MOTION", lw=0)
    ax.set_ylim(ylo, yhi)
    ax.set(ylabel="activity (log)", title="Step 9 — decision with threshold + hysteresis")
    ax.legend(loc="upper left", fontsize=8)
    for t0, t1, lab in segments:
        axt.axvspan(t0, t1, color=ORANGE if lab == "walking" else GRID, alpha=0.6 if lab == "walking" else 1, lw=0)
        axt.text((t0 + t1) / 2, 0.5, lab, ha="center", va="center", color=INK)
    axt.set(yticks=[], xlabel="time (s)", ylabel="truth")
    axt.grid(False)
    axt.set_xlim(st.t_act[0], st.t_act[-1])
    _finish(fig, out / "step9_decision.png", synthetic)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--empty", type=Path, required=True)
    ap.add_argument("--walking", type=Path, required=True)
    ap.add_argument("--mixed", type=Path, help="session with labelled segments (for step 9)")
    ap.add_argument("--calib", type=Path, help="calibration .json (threshold + config)")
    ap.add_argument("--out", type=Path, default=Path("../figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    threshold, cfg = load_calibration(args.calib) if args.calib else (None, PipelineConfig())
    rec_e, rec_w = load_recording(args.empty), load_recording(args.walking)
    synthetic = bool(rec_e.meta.get("synthetic") or rec_w.meta.get("synthetic"))
    st_e, st_w = run_offline(rec_e.main_link, cfg, threshold), run_offline(rec_w.main_link, cfg, threshold)

    step1_timing(rec_w.main_link.t, args.out, synthetic)
    step4_heatmaps(st_e, st_w, args.out, synthetic)
    step5_hampel(st_w, args.out, synthetic)
    step6_response(cfg, args.out, synthetic)
    step6_spectra(st_e, st_w, cfg, args.out, synthetic)
    step6_zero_phase(st_w, cfg, args.out, synthetic)
    step7_pca(st_e, st_w, cfg, args.out, synthetic)
    curves = [("empty", st_e), ("walking", st_w)]
    if args.mixed:
        rec_m = load_recording(args.mixed)
        st_m = run_offline(rec_m.main_link, cfg, threshold)
        curves.append(("mixed", st_m))
        if threshold is not None:
            step9_decision(st_m, label_segments(rec_m, rec_m.main_link), threshold, args.out, synthetic)
    step8_activity(curves, threshold, args.out, synthetic)


if __name__ == "__main__":
    main()
