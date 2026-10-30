"""Run from python/:  python -m pytest -q"""
from pathlib import Path

import numpy as np
import pytest

from capture.simulate import simulate_session
from pipeline import filters
from pipeline.core import PipelineConfig, StreamingPipeline, run_offline, threshold_from_activity
from pipeline.hampel import StreamingHampel, hampel
from pipeline.parse import DATA_SUBCARRIERS, load_recording, parse_line, packets_to_links, unwrap_timestamps
from pipeline.resample import StreamingResampler, resample_uniform

HERE = Path(__file__).parent
REAL_SAMPLE = HERE / "esp32_readme_sample.csv"   # real ESP32 output from Espressif's README


def test_real_sample_parses_to_52_smooth_subcarriers():
    rec = load_recording(REAL_SAMPLE)
    link = rec.main_link
    assert link.amp.shape == (14, 52)
    # with the right byte layout every data subcarrier carries signal (no guard zeros)
    # and the amplitude changes smoothly from one subcarrier to the next
    assert link.amp.min() > 3
    assert np.median(np.abs(np.diff(link.amp, axis=1))) < 1.5


def test_imag_first_then_real():
    meta = "CSI_DATA,0,aa:bb:cc:dd:ee:ff,-30,11,1,6,1,0,1,0,1,0,0,-93,0,13,2,100,0,67,0,128,0"
    buf = [0] * 128
    buf[2 * 38], buf[2 * 38 + 1] = 3, 4        # entry 38 -> k = 6 (ascending order): imag 3, real 4
    pkt = parse_line(meta + ',"[' + ",".join(map(str, buf)) + ']"')
    assert pkt.csi[38] == 4 + 3j


@pytest.mark.parametrize("sec", [0, 2])
def test_simulated_layouts_roundtrip(tmp_path, sec):
    lines, _ = simulate_session([(0, 5, "empty")], 5, seed=1, secondary_channel=sec)
    f = tmp_path / "s.csv"
    f.write_text("\n".join(lines))
    amp = load_recording(f).main_link.amp
    assert amp.shape[1] == len(DATA_SUBCARRIERS) and amp.min() > 1


def test_broken_lines_are_skipped():
    assert parse_line("I (331) wifi:wifi driver task") is None
    assert parse_line('CSI_DATA,1,aa,-3,11,1,6,1,0,1,0,1,0,0,-93,0,13,2,5,0,67,0,128,1,"[1,2,3') is None


def test_timestamp_wrap():
    ts = np.array([2**32 - 20_000, 2**32 - 10_000, 0, 10_000])
    assert np.all(np.diff(unwrap_timestamps(ts)) == 10_000)


def test_resample_and_gap_mask():
    t = np.r_[np.arange(0, 1, 0.01), np.arange(1.3, 2, 0.01)]
    x = np.column_stack([t, 2 * t])
    grid, xu, gaps = resample_uniform(t, x, 100)
    assert np.allclose(xu[:, 0], grid) and np.allclose(xu[:, 1], 2 * grid)
    assert gaps[(grid > 1.0) & (grid < 1.29)].all() and not gaps[grid < 0.98].any()


def test_streaming_resampler_matches_offline():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.005, 0.015, 300))
    x = rng.normal(size=(300, 3))
    grid, xu, _ = resample_uniform(t, x, 100)
    r = StreamingResampler(100)
    out = [s for ti, xi in zip(t, x) for _, s, _ in r.push(ti, xi)]
    assert np.allclose(np.array(out), xu[:len(out)])


def test_hampel_removes_spike_keeps_rest():
    x = np.sin(np.linspace(0, 6, 200))
    y = x.copy(); y[100] += 5
    cleaned, mask = hampel(y, 11, 3)
    assert mask[100] and abs(cleaned[100] - x[100]) < 0.1
    assert mask.sum() <= 3


def test_streaming_hampel_matches_offline():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 4))
    off, _ = hampel(x, 11, 3)
    h = StreamingHampel(11, 3)
    live = np.vstack([h.push(x[i:i + 37]) for i in range(0, 500, 37)])
    assert np.allclose(live, off[5:5 + len(live)])


def test_butterworth_minus_3db_at_cutoff():
    sos = filters.butter_lowpass(4, 20, 100)
    f, mag, _ = filters.frequency_response(sos, 100, 4096)
    assert abs(np.interp(20, f, mag) + 3.01) < 0.05


def test_streaming_filter_matches_causal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 3)) + 5
    sos = filters.butter_lowpass(4, 20, 100)
    whole = filters.apply_causal(sos, x)
    sf = filters.StreamingFilter(sos)
    chunks = np.vstack([sf.push(x[i:i + 50]) for i in range(0, 400, 50)])
    assert np.allclose(whole, chunks)


@pytest.fixture(scope="module")
def sim_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    out = {}
    for name, segs, seed in [("calib", [(0, 60, "empty")], 1), ("empty", [(0, 60, "empty")], 2),
                             ("walk", [(0, 60, "walking")], 3)]:
        lines, _ = simulate_session(segs, 60, seed=seed)
        (d / f"{name}.csv").write_text("\n".join(lines))
        out[name] = d / f"{name}.csv"
    return out


def test_end_to_end_detection(sim_files):
    cfg = PipelineConfig()
    st = run_offline(load_recording(sim_files["calib"]).main_link, cfg)
    thr = threshold_from_activity(st.activity[st.valid], cfg)
    e = run_offline(load_recording(sim_files["empty"]).main_link, cfg, thr).decisions.motion
    w = run_offline(load_recording(sim_files["walk"]).main_link, cfg, thr).decisions.motion
    assert e.mean() < 0.1 and w.mean() > 0.9


def test_streaming_pipeline_matches_offline_causal(sim_files):
    cfg = PipelineConfig(zero_phase=False)
    link = load_recording(sim_files["walk"]).main_link
    off = run_offline(link, cfg)
    sp = StreamingPipeline(cfg)
    outs = [o for t, a in zip(link.t, link.amp) for o in sp.push_packet(t, a)]
    live = np.array([o.activity for o in outs])
    # live lags offline by the Hampel delay (5 samples) and starts its filter later,
    # so compare the overall level rather than sample by sample
    n = min(len(live), len(off.activity))
    assert n > 80
    ratio = np.median(live[:n]) / np.median(off.activity[:n])
    assert 0.7 < ratio < 1.4
