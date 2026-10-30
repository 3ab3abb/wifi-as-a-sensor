# Python — capture, DSP pipeline, evaluation, live demo

Run everything from this `python/` folder (`cd python`). Requirements: `pip install -r requirements.txt`
(all already installed in the Anaconda Python on this laptop).

| Folder | What | Pipeline step |
|---|---|---|
| `capture/sources.py` | swappable data sources: `SerialSource` (ESP32), `ReplaySource` (file) | 1–3 |
| `capture/record.py` | record a labelled session to `data/raw/` | 3 |
| `capture/simulate.py` | synthetic CSI in the exact ESP32 format (test before hardware) | — |
| `pipeline/parse.py` | text → complex CSI → 52 subcarrier amplitudes, per link | 4 |
| `pipeline/resample.py` | per-packet AGC normalization, uniform 100 Hz grid, timing stats | 1, 4 |
| `pipeline/hampel.py` | Hampel outlier filter (offline + streaming) | 5 |
| `pipeline/filters.py` | Butterworth design, zero-phase vs causal, frequency response | 6 |
| `pipeline/combine.py` | background removal + PCA (via SVD) | 7 |
| `pipeline/measure.py` | moving variance of top components = activity | 8 |
| `pipeline/detect.py` | threshold calibration, hysteresis, presence hold | 9 |
| `pipeline/core.py` | `PipelineConfig` (all parameters), `run_offline`, `StreamingPipeline` | 4–9 |
| `analysis/evaluate.py` | accuracy / false alarms / misses / confusion matrix | 10 |
| `analysis/checkpoints.py` | one plot per step → `figures/` → slides | all |
| `live/live_detect.py` | real-time demo window | 10 |
| `tests/` | `python -m pytest -q` | — |

## Without the ESP32 (works today)

```bash
python -m capture.simulate --out ../data/synthetic
python -m analysis.evaluate ../data/synthetic --calib ../data/synthetic/sim_empty_1.csv \
       --save-calib ../data/processed/calibration_synthetic.json
python -m analysis.checkpoints --empty ../data/synthetic/sim_empty_2.csv \
       --walking ../data/synthetic/sim_walking_1.csv --mixed ../data/synthetic/sim_mixed_1.csv \
       --calib ../data/processed/calibration_synthetic.json --out ../figures/synthetic
python -m live.live_detect --replay ../data/synthetic/sim_mixed_1.csv --calibrate 60
```

## With the ESP32 (after flashing, see `firmware/README.md`)

Placement: ESP32 and router 3–5 m apart, chest height, don't move them between sessions.

```bash
# 1. record labelled sessions (2 min each). Leave the room for "empty" ones.
python -m capture.record --label empty   --placement room1-4m   # at least 2 of these
python -m capture.record --label walking --placement room1-4m
python -m capture.record --label sitting --placement room1-4m
python -m capture.record --label empty   --placement room1-4m   # mixed: type "walking"+Enter when
                                                                 # you enter, "empty"+Enter when you leave
# 2. evaluate (calibrate on one empty session, test on the rest)
python -m analysis.evaluate ../data/raw --calib ../data/raw/<first_empty>.csv \
       --save-calib ../data/processed/calibration.json
# 3. checkpoint figures for the slides
python -m analysis.checkpoints --empty ... --walking ... --mixed ... \
       --calib ../data/processed/calibration.json --out ../figures
# 4. live demo (calibrate with the causal filter, as used live)
python -m analysis.evaluate ../data/raw --calib ../data/raw/<first_empty>.csv --causal \
       --save-calib ../data/processed/calibration_live.json
python -m live.live_detect --calib ../data/processed/calibration_live.json
```

## Design notes
* **Swappable source:** everything downstream sees text lines in the ESP32 format. Phase 2
  (ESP-NOW) only needs a different firmware; the parser groups packets by transmitter MAC, so
  several links are already supported (`Recording.links`).
* **Same maths offline and live:** PCA is fitted on a trailing 10 s history and activity is the
  variance over the last 1 s, so the offline evaluation measures what the live demo does.
  `PipelineConfig(zero_phase=False)` evaluates with the causal filter used live.
* **Packet gaps:** windows that contain a hole of > 50 ms between packets are marked invalid and
  don't change the decision — straight-line interpolation over a hole looks like fake activity.
