# Architecture

## Data flow

```
  Proteus VSM bench                    simulate_collar.py
  (ATmega328P + LM35 + POT             (seeded generator, 60 animals,
   + COMPIM NMEA feed)                  14 days, disclosed parameters)
          |                                       |
          |  SMARTHERD,<id>,<utc>,<lat>,<lon>,        |
          |  <temp>,<hr>,<fix>                     |
          v                                        v
   proteus_bridge.py  -----------------------> build_features()
   (feed / read / replay)                     10 channels x 24 samples
                                              (2-hour window, 30-min stride)
                                                     |
                     +-------------------------------+
                     |                               |
                     v                               v
          deterministic fence check          ml/model.py  CNN-GRU
          (fence_dist < 0)                   33,382 params
                     |                               |
                     +-------------> ml/risk_engine.py <-------+
                                            |                  |
                                            |    data/sadc_animal_diseases_2011.csv
                                            v
                                  routed alert:
                                  severity, channels (app/sms/dvs),
                                  action text, ranked disease shortlist
```

## The ten channels

| # | Channel | Origin |
|---|---|---|
| 0 | `speed_ms` | Haversine step / 300 s |
| 1 | `step_m` | Haversine between consecutive fixes |
| 2 | `turn_rad` | wrapped bearing difference |
| 3 | `herd_dist_m` | distance to herd centroid — badisa indicator 1, *go sala morago* |
| 4 | `fence_dist_m` | signed; positive inside |
| 5 | `temp_c` | body temperature |
| 6 | `hr_bpm` | heart rate |
| 7 | `activity_idx` | rolling kinematic energy, accelerometer proxy |
| 8 | `hour_sin` | time of day |
| 9 | `hour_cos` | time of day |

The movement-only ablation drops channels 5, 6 and 7.

## Model

```
Conv1d(10 -> 32, k=5, pad=2) + BatchNorm + ReLU
Conv1d(32 -> 64, k=3, pad=1) + BatchNorm + ReLU
MaxPool1d(2)                            24 -> 12 timesteps
Dropout(0.15)
GRU(64 -> 64, 1 layer, unidirectional)
Attention pooling over the 12 GRU outputs
Dropout(0.3)
Linear(64 -> 5)
```

**Why convolution then recurrence.** The convolutions learn short local
motifs — a fever ramp, a hesitant gait, an acceleration burst — without the
recurrent layer having to remember them across the whole window. The GRU then
reads the *sequence* of motifs. That split matters because these classes are
separated by order, not by level: resting and sick have the same mean speed,
grazing and fleeing both show high speed. Only temporal shape separates them.

**Why GRU over LSTM.** About 25% fewer parameters for the same job. The
eventual target is an ESP32-class device, where every kilobyte of flash is a
battery-life decision.

**Why attention pooling rather than the last hidden state.** A two-hour window
is usually mostly normal with a short informative stretch inside it. Taking
the final hidden state throws away the part that matters.

## Training protocol

- Splits **by animal**, 70/10/20. Consecutive windows overlap in time, so a
  random window split would place near-duplicates in both train and test and
  report an accuracy nobody can reproduce.
- Normalisation fitted on training animals only.
- Class-weighted cross-entropy. Theft is 1.2% of windows; unweighted loss
  learns to ignore it.
- AdamW, lr 2e-3, cosine schedule, gradient clipping at 3.0, 25 epochs.
- Model selected on best validation macro-F1, then evaluated once on test.

## Two rules that are not negotiable

**Separate failure domains.** Position recording and alert logic fail
independently. A bad fence configuration or a model error can never blank the
map or silence the geofence alarm. The model escalates; it does not gate.

**Every number is written by the code that computed it.** `ml/train.py`
serialises everything into `ml/artifacts/metrics.json`, and `make_figures.py`
reads only from that file. No figure can disagree with a reported number, and
no reported number can be typed by hand.
