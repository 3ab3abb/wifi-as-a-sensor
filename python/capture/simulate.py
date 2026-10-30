"""Synthetic CSI generator — lets us build and test the whole pipeline before the ESP32
is available. Output files use EXACTLY the csi_recv_router serial format, so the parser,
pipeline, evaluation and live demo (replay mode) run unchanged on real data later.

!! Figures made from this data are for testing only, not results. !!

Physical model (multipath channel = sum of echoes, see the "room is an LTI system" slide):
    H(f_k, t) = sum_p  a_p * exp(-j 2 pi (fc + f_k) d_p(t) / c)
* static paths: line of sight (LOS) TX->RX plus wall/furniture reflections (fixed d_p)
* person: one extra reflected path TX -> person -> RX whose length changes as they move,
  and they partly block the LOS when standing near the TX-RX line
* sitting: the person is fixed, only breathing (a few mm at ~0.25 Hz) and rare fidgets
Plus the real-world impairments the pipeline must handle: thermal noise, int8
quantization, AGC gain steps, random phase offsets (unsynchronized clocks), interference
outliers, timing jitter, lost packets and long gaps.

Usage (from the python/ folder):
    python -m capture.simulate --out ../data/synthetic          # standard test set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.parse import _K_ASCENDING, _K_NONE, DATA_SUBCARRIERS, DEFAULT_HEADER

C = 3e8
FC = 2.437e9                      # Wi-Fi channel 6
DF = 312.5e3                      # subcarrier spacing
TX, RX = np.array([0.0, 0.0]), np.array([4.0, 0.0])   # 4 m apart
ROOM = (-1.0, 5.0, -2.5, 2.5)     # x_min, x_max, y_min, y_max
MAC = "aa:bb:cc:00:00:01"


def room_paths(rng):
    """Static multipath: LOS + 7 reflections (fixed for a given room seed)."""
    d_los = np.linalg.norm(RX - TX)
    d = np.r_[d_los, d_los + rng.uniform(1.0, 14.0, 7)]
    a = np.r_[1.0, rng.uniform(0.15, 0.6, 7) * d_los / d[1:]] * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    return d, a


def walk_trajectory(t, rng, speed=1.0):
    """Random waypoint walk inside the room, smooth enough for a human."""
    pos = np.empty((t.size, 2))
    p = rng.uniform([ROOM[0], ROOM[2]], [ROOM[1], ROOM[3]])
    target = p
    v = speed * rng.uniform(0.7, 1.3)
    for i in range(t.size):
        dt = t[i] - t[i - 1] if i else 0.0
        step = target - p
        dist = np.linalg.norm(step)
        if dist < 0.1:
            target = rng.uniform([ROOM[0] + 0.3, ROOM[2] + 0.3], [ROOM[1] - 0.3, ROOM[3] - 0.3])
            v = speed * rng.uniform(0.7, 1.3)
        else:
            p = p + step / dist * min(v * dt, dist)
        pos[i] = p
    return pos


def sitting_trajectory(t, rng):
    """Fixed seat + breathing (chest moves ~4 mm) + a short fidget every ~15 s."""
    seat = np.array([rng.uniform(1.0, 3.0), rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)])
    f_br = rng.uniform(0.2, 0.3)                       # 12-18 breaths/min
    toward = (seat - (TX + RX) / 2) / np.linalg.norm(seat - (TX + RX) / 2)
    disp = 0.004 * np.sin(2 * np.pi * f_br * t)
    for t0 in np.cumsum(rng.exponential(15.0, 20)):
        m = (t >= t0) & (t < t0 + 0.6)
        disp[m] += 0.03 * np.sin(np.pi * (t[m] - t0) / 0.6)
    return seat + disp[:, None] * toward, f_br


def packet_times(duration, rng, rate=100.0):
    n = int(duration * rate * 1.1)
    dt = np.clip(1 / rate + rng.normal(0, 0.0015, n), 0.002, None)
    busy = rng.random(n) < 0.002                       # router busy: 100-300 ms hole
    dt[busy] += rng.uniform(0.1, 0.3, busy.sum())
    t = np.cumsum(dt)
    t = t[t < duration]
    return t[rng.random(t.size) > 0.03]                # 3% packets lost


def simulate_session(segments, duration, room_seed=1, seed=0, secondary_channel=0):
    """segments: list of (t_start, t_end, label) with label in {empty, walking, sitting}.
    Returns (lines, info)."""
    rng = np.random.default_rng(seed)
    d_static, a_static = room_paths(np.random.default_rng(room_seed))
    t = packet_times(duration, rng)
    f = FC + DATA_SUBCARRIERS * DF                     # 52 subcarrier frequencies

    # slow environmental drift of the static paths (temperature, etc.)
    drift = 1 + 0.003 * np.sin(2 * np.pi * rng.uniform(0.005, 0.02) * t[:, None] + rng.uniform(0, 6, 8))
    H = (drift * a_static) @ np.exp(-2j * np.pi * np.outer(d_static, f) / C)

    info = {"breathing_hz": None}
    for t0, t1, label in segments:
        m = (t >= t0) & (t < t1)
        if label == "empty" or not m.any():
            continue
        if label == "walking":
            pos = walk_trajectory(t[m], rng)
        else:
            pos, info["breathing_hz"] = sitting_trajectory(t[m], rng)
        d_p = np.linalg.norm(pos - TX, axis=1) + np.linalg.norm(pos - RX, axis=1)
        a_p = 0.35 * np.linalg.norm(RX - TX) / d_p * np.exp(1j * rng.uniform(0, 2 * np.pi))
        H[m] += a_p[:, None] * np.exp(-2j * np.pi * np.outer(d_p, f) / C)
        # LOS blockage when the person is close to the TX-RX line (between the devices)
        dist_line = np.where((pos[:, 0] > TX[0]) & (pos[:, 0] < RX[0]), np.abs(pos[:, 1]), np.inf)
        block = 0.5 * np.exp(-(dist_line / 0.3) ** 2)
        los = a_static[0] * np.exp(-2j * np.pi * f * d_static[0] / C)
        H[m] -= block[:, None] * los[None, :]

    # receiver impairments
    sig = np.sqrt(np.mean(np.abs(H) ** 2))
    H = H + (sig / 10 ** (25 / 20)) * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)) / np.sqrt(2)
    burst = rng.random(t.size) < 0.005                 # interference outliers
    for i in np.flatnonzero(burst):
        sel = rng.random(H.shape[1]) < 0.3
        H[i, sel] *= rng.uniform(0.3, 3.0, sel.sum())
    step_times = np.cumsum(rng.exponential(15.0, 50))  # AGC gain steps of +-2 dB
    gain_db = np.r_[0, rng.uniform(-2, 2, 50)][np.searchsorted(step_times, t)]
    H = H * 10 ** (gain_db / 20)[:, None]
    phase = rng.uniform(0, 2 * np.pi, t.size)[:, None] + rng.normal(0, 0.05, t.size)[:, None] * DATA_SUBCARRIERS
    H = H * np.exp(1j * phase)                          # unsynchronized clocks -> random phase
    scale = 18 / np.mean(np.abs(H))                     # like real ESP32 values (~5-30)
    q = np.round(H * scale)
    re = np.clip(q.real, -128, 127).astype(int)
    im = np.clip(q.imag, -128, 127).astype(int)

    # pack into 64 raw entries in ESP32 order (imag, real), guards/DC = 0
    k_of_entry = _K_NONE if secondary_channel == 0 else _K_ASCENDING
    pos_of_k = {k: j for j, k in enumerate(DATA_SUBCARRIERS)}
    ts0 = int(rng.integers(0, 2**32))                   # random start: exercises the wrap
    ts = (ts0 + np.round(t * 1e6).astype(np.int64)) % 2**32
    rssi = np.round(-45 + 20 * np.log10(np.mean(np.abs(H), axis=1) / np.mean(np.abs(H)))).astype(int)
    lines = [",".join(DEFAULT_HEADER)]
    for i in range(t.size):
        buf = []
        for k in k_of_entry:
            j = pos_of_k.get(k)
            buf += [0, 0] if j is None else [im[i, j], re[i, j]]
        buf[:4] = rng.integers(-60, 100, 4).tolist()    # first word invalid (ESP32 quirk)
        meta = [ "CSI_DATA", i, MAC, rssi[i], 11, 0, 0, 0, 0, 1, 0, 0, 0, 0, -93, 0, 6,
                secondary_channel, ts[i], 0, 67, 0, 128, 1]
        lines.append(",".join(map(str, meta)) + ',"[' + ",".join(map(str, buf)) + ']"')
    return lines, info


STANDARD_SET = [
    # name, segments (start, end, label), duration
    ("empty_1", [(0, 120, "empty")], 120),
    ("empty_2", [(0, 120, "empty")], 120),
    ("empty_3", [(0, 120, "empty")], 120),
    ("walking_1", [(0, 120, "walking")], 120),
    ("walking_2", [(0, 120, "walking")], 120),
    ("sitting_1", [(0, 120, "sitting")], 120),
    ("mixed_1", [(0, 40, "empty"), (40, 80, "walking"), (80, 120, "empty")], 120),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../data/synthetic")
    ap.add_argument("--room-seed", type=int, default=1)
    ap.add_argument("--secondary-channel", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed, (name, segs, dur) in enumerate(STANDARD_SET, start=100):
        lines, info = simulate_session(segs, dur, args.room_seed, seed, args.secondary_channel)
        (out / f"sim_{name}.csv").write_text("\n".join(lines) + "\n")
        labels = {lab for _, _, lab in segs}
        meta = {"label": labels.pop() if len(labels) == 1 else "mixed",
                "segments": [list(s) for s in segs], "synthetic": True, "seed": seed,
                "room_seed": args.room_seed, "placement": "sim: TX-RX 4 m", **info}
        (out / f"sim_{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out / f'sim_{name}.csv'}  ({len(lines) - 1} packets)")


if __name__ == "__main__":
    main()
