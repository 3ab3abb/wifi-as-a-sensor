"""Step 3 — Record a labelled session from the ESP32 to data/raw/.

Each session gives two files:
    data/raw/<date>_<time>_<label>_<placement>.csv   raw serial lines, prefixed by laptop time
    data/raw/<date>_<time>_<label>_<placement>.json  label, placement, notes, markers

Markers: while recording, type a label and press Enter (e.g. `walking` then later `empty`)
to mark the moment the situation changed — that gives per-segment ground truth for
mixed sessions. The session's initial label counts as the first marker.

Usage (from python/):
    python -m capture.record --label empty --placement livingroom-4m --duration 120
    python -m capture.record --label empty --duration 120 --replay ../data/synthetic/sim_mixed_1.csv  # test without ESP32
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from pipeline.parse import parse_line
from capture.sources import ReplaySource, SerialSource

LABELS = ("empty", "walking", "sitting")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, choices=LABELS, help="situation at the start")
    ap.add_argument("--placement", default="default", help="short tag for device placement, e.g. livingroom-4m")
    ap.add_argument("--duration", type=float, default=120, help="seconds (Ctrl+C stops earlier)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--port", help="serial port (auto-detected if omitted)")
    ap.add_argument("--replay", type=Path, help="read from a file instead of the ESP32 (testing)")
    ap.add_argument("--out", type=Path, default=Path("../data/raw"))
    args = ap.parse_args()

    src = ReplaySource(args.replay) if args.replay else SerialSource(args.port)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / f"{stamp}_{args.label}_{args.placement}"
    markers = [[time.time(), args.label]]

    def read_markers():
        for line in sys.stdin:
            lab = line.strip()
            if lab in LABELS:
                markers.append([time.time(), lab])
                print(f"  marker: {lab}")
            elif lab:
                print(f"  unknown label {lab!r}; use one of {LABELS}")

    threading.Thread(target=read_markers, daemon=True).start()
    print(f"recording {args.duration:.0f} s -> {base}.csv   (type a label + Enter to mark a change)")

    n_ok = n_bad = 0
    t_end = time.time() + args.duration
    last_print = 0.0
    try:
        with open(f"{base}.csv", "w") as f:
            for host_t, line in src:
                if line.startswith("CSI_DATA"):
                    if parse_line(line) is not None:
                        f.write(f"{host_t:.6f},{line}\n")
                        n_ok += 1
                    else:
                        n_bad += 1
                if host_t - last_print > 2:
                    last_print = host_t
                    print(f"  {n_ok} packets ({n_ok / max(host_t - (t_end - args.duration), 1e-9):.0f}/s), {n_bad} broken lines")
                if host_t >= t_end:
                    break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        src.close()

    meta = {"label": args.label if len({m[1] for m in markers}) == 1 else "mixed",
            "markers": markers, "placement": args.placement, "notes": args.notes,
            "source": str(args.replay) if args.replay else getattr(src, "port", "serial"),
            "packets": n_ok, "broken_lines": n_bad, "started": stamp}
    Path(f"{base}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {n_ok} packets to {base}.csv (+ .json)")


if __name__ == "__main__":
    main()
