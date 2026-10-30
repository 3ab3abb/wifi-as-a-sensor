"""Step 10 — Evaluate the detector on labelled recordings.

Protocol (so the numbers are honest):
* The threshold is calibrated on ONE empty-room recording (the calibration session),
  which is then NOT used for testing — testing on the data you calibrated on would
  flatter the false-alarm rate.
* Every other recording is cut into decision windows (one per hop); each window gets
  the ground-truth label of its time (whole-session label, or per-segment labels from
  the .json sidecar).
* Motion detection is a yes/no question: truth "motion" = walking, "no motion" = empty.
  Sitting sessions are reported separately (is a sitting person detected as present?).
* Windows within `margin_s` of a label change are skipped (the exact moment someone
  "starts walking" is fuzzy).

Usage (from python/):
    python -m analysis.evaluate ../data/synthetic --calib ../data/synthetic/sim_empty_1.csv
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline.core import PipelineConfig, run_offline, save_calibration, threshold_from_activity
from pipeline.parse import LinkData, Recording, load_recording

MOTION_LABELS = {"walking"}
NO_MOTION_LABELS = {"empty"}


def label_segments(rec: Recording, link: LinkData) -> list[tuple[float, float, str]]:
    """Ground truth as (start_s, end_s, label) in the link's time base (s since 1st packet)."""
    meta = rec.meta
    if "segments" in meta:
        return [tuple(s) for s in meta["segments"]]
    if "markers" in meta and np.isfinite(link.host_t).any():
        # markers from capture.record: [host_time, label]; convert host time -> link time
        h = link.host_t
        to_link = lambda ht: float(np.interp(ht, h, link.t))
        marks = sorted(meta["markers"])
        out = []
        for i, (ht, lab) in enumerate(marks):
            end = to_link(marks[i + 1][0]) if i + 1 < len(marks) else link.t[-1] + 1
            out.append((to_link(ht), end, lab))
        return out
    if "label" in meta:
        return [(0.0, float(link.t[-1]) + 1, meta["label"])]
    raise ValueError(f"{rec.path}: no label/segments/markers in the .json sidecar")


def window_labels(t_act: np.ndarray, segments, margin_s: float = 2.0) -> np.ndarray:
    """Label of each decision window (by its end time); '' near a label change."""
    labels = np.full(t_act.size, "", dtype=object)
    for t0, t1, lab in segments:
        labels[(t_act >= t0 + margin_s) & (t_act < t1 - margin_s)] = lab
    if len(segments) == 1:   # a single-label session has no change to blur
        t0, t1, lab = segments[0]
        labels[(t_act >= t0) & (t_act < t1)] = lab
    return labels


def confusion(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum(truth & pred)); tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred)); fn = int(np.sum(truth & ~pred))
    n = tp + tn + fp + fn
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / n if n else np.nan,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
            "miss_rate": fn / (fn + tp) if fn + tp else np.nan}


def calibrate(calib_path: Path, cfg: PipelineConfig) -> float:
    st = run_offline(load_recording(calib_path).main_link, cfg)
    return threshold_from_activity(st.activity[st.valid], cfg)


def evaluate(paths: list[Path], threshold: float, cfg: PipelineConfig, margin_s: float = 2.0,
             use: str = "motion") -> dict:
    """use = 'motion' (after hysteresis) or 'raw' (threshold only)."""
    all_truth, all_pred, per_session, sitting = [], [], [], []
    for p in paths:
        rec = load_recording(p)
        link = rec.main_link
        st = run_offline(link, cfg, threshold)
        labels = window_labels(st.t_act, label_segments(rec, link), margin_s)
        pred = getattr(st.decisions, use)
        m = np.isin(labels, list(MOTION_LABELS | NO_MOTION_LABELS))
        truth = np.isin(labels[m], list(MOTION_LABELS))
        all_truth.append(truth); all_pred.append(pred[m])
        row = {"file": p.name, "label": rec.meta.get("label", "?"), "windows": int(m.sum()),
               "detected_motion_frac": float(pred[m].mean()) if m.any() else np.nan,
               "invalid_frac": float(1 - st.valid.mean())}
        s = labels == "sitting"
        if s.any():
            sitting.append({"file": p.name, "motion_frac": float(st.decisions.motion[s].mean()),
                            "present_frac": float(st.decisions.present[s].mean())})
        per_session.append(row)
    truth, pred = np.concatenate(all_truth), np.concatenate(all_pred)
    return {"threshold": threshold, "overall": confusion(truth, pred),
            "sessions": per_session, "sitting": sitting}


def print_report(res: dict):
    o = res["overall"]
    print(f"threshold = {res['threshold']:.4g}")
    print(f"accuracy {o['accuracy']:.3f} | false alarms {o['false_alarm_rate']:.3f} | misses {o['miss_rate']:.3f}")
    print("confusion matrix (rows = truth, cols = predicted):")
    print(f"               no-motion  motion\n  no-motion  {o['tn']:9d} {o['fp']:7d}\n  motion     {o['fn']:9d} {o['tp']:7d}")
    print("per session:")
    for r in res["sessions"]:
        print(f"  {r['file']:28s} {r['label']:8s} windows={r['windows']:4d} "
              f"motion={r['detected_motion_frac']:.2f} invalid={r['invalid_frac']:.2f}")
    for s in res["sitting"]:
        print(f"  sitting {s['file']}: motion {s['motion_frac']:.2f}, present {s['present_frac']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="empty-room recording used for the threshold")
    ap.add_argument("--save-calib", type=Path, help="write threshold + config to this .json")
    ap.add_argument("--causal", action="store_true", help="evaluate with the causal (live) filter")
    args = ap.parse_args()
    cfg = PipelineConfig(zero_phase=not args.causal)
    thr = calibrate(args.calib, cfg)
    tests = sorted(p for p in args.folder.glob("*.csv") if p.resolve() != args.calib.resolve())
    res = evaluate(tests, thr, cfg)
    print_report(res)
    if args.save_calib:
        save_calibration(args.save_calib, thr, cfg, calibrated_on=str(args.calib))


if __name__ == "__main__":
    main()
