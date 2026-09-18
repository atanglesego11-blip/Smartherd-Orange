# SmartHerd.ai

**A digital *modisa* for Botswana.**
Open Source Hackathon 2026 — Open Source Community Botswana / Orange Digital Center / UniPod / University of Botswana.

SmartHerd.ai watches a herd the way a *modisa* does — for the animal that lags
behind, the one standing apart, the one that leaves in a straight line at
night — and it does it at three in the morning, for two hundred animals, in
a country where the herder who could do that is increasingly not there.

Themes: **Indigenous Knowledge & Resource Protection** (primary), **Defence &
Security**, **Health & Wellness**.

Licence: MIT. No physical hardware is required to run any of this.

---

## What it actually does

A collar reports position, body temperature and heart rate every five
minutes. SmartHerd.ai turns that stream into three things a farmer can act on:

1. **Is this animal getting sick?** A 1D CNN-GRU classifies two-hour
   telemetry windows. When it flags illness, a risk engine joins that flag to
   the SADC animal-disease record and returns a ranked shortlist of what to
   rule out — with each condition's regional outbreak burden, its Botswana
   wildlife hosts, and whether it can infect people. A zoonotic hit escalates
   the alert to the Department of Veterinary Services, not just to the
   farmer's phone.
2. **Is this animal being stolen?** Sustained, directed, nocturnal movement
   away from the kraal, with the animal separated from the herd.
3. **Is something in the grazing area?** A sudden scatter with a heart-rate
   spike and no fever.

And one thing it deliberately does *not* do: the geofence breach alarm is
plain arithmetic, it runs whether the model agrees or not, and **the model
can raise its priority but can never cancel it.** A machine-learning false
negative must not be able to silence a deterministic alarm.

## Why this is an indigenous-knowledge project, not a sensor project

Every channel the model reads was chosen because a herder already uses it.
`herd_dist_m` exists because *go sala morago* — going behind — is the first
sign badisa look for. `turn_rad` variance exists because animals do not walk
in straight lines at night of their own accord.

The mapping is written down in
[`data/indigenous_knowledge/badisa_indicators.md`](data/indigenous_knowledge/badisa_indicators.md),
with an honest status column marking which indicators have been validated
with practising herders and which are still our reading of the literature,
plus a list of signs we know matter and **cannot yet sense** — rumination
being the big one. A system that only implements the knowledge it finds
convenient is not preserving anything.

Indicator 10 is the one to read: *a whole herd moving fast in daylight is
normal — it is being driven to the dip tank.* Kinematically that is identical
to rustling. A system built without that knowledge sends a theft alarm every
dipping day, the farmer stops reading the alarms, and the one real theft goes
unanswered. That distinction is not in any dataset you can download. It came
from asking.

## Results

Trained on 40,140 windows from 60 simulated animals over 14 days, split **by
animal** so no window leaks between train and test. Held-out test set: 8,028
windows.

| Class | CNN-GRU, movement + physiology | CNN-GRU, movement only | Random forest |
|---|---|---|---|
| grazing | 0.979 | 0.972 | 0.965 |
| resting | 0.988 | 0.963 | 0.980 |
| **illness_suspect** | **0.993** | **0.911** | 0.980 |
| theft_suspect | 0.991 | 0.982 | 0.987 |
| predator_flight | 0.950 | 0.952 | 0.988 |
| **macro F1** | **0.980** | 0.956 | 0.980 |

Model: 33,382 parameters, 68 seconds to train on one CPU core.

**The column that matters is illness.** Remove temperature and heart rate and
illness detection falls from 0.993 to 0.911, because a sick animal standing
still and a healthy animal in midday shade are the same thing in movement
space. That gap is the argument for physio-spatial fusion, and it is
computed, not asserted.

**Three things we are not hiding.** The random forest baseline matches the
deep model on overall macro-F1 and beats it on predator flight — on a
synthetic benchmark this easy, a classical method is competitive, and we
report it. The absolute numbers are high because the benchmark is easy; the
*gap* between conditions is the result, not the level. And every figure here
is computed on synthetic data, which brings us to the next section.

## Honesty about the data

**All numbers in this repository come from simulated animals.** There is no
field data in this build and we do not pretend otherwise.

The generator (`simulator/simulate_collar.py`) is fully disclosed, seeded at
42, and grounded in three public sources: the SADC animal-disease clinical
sign strings, published small-stock physiological baselines, and the herder
indicators above. It is also deliberately hard. Our first run scored 99% on
everything, which meant the task was fake, so we rebuilt it with the three
confounders that make this difficult in real life:

- **midday shade rest** — healthy animals that stop moving for hours in the
  heat, which is movement-identical to illness
- **legitimate herd drives** — dip-tank days, fast and directed and in
  daylight, which is kinematically identical to rustling
- **subclinical illness** — a third of episodes carry little or no fever

Sick animals in the simulator lag the herd but do not wander off, because
real ones don't; letting them drift would have handed the classifier a free
signal and made the movement-only ablation look far better than it deserves.

These figures measure whether the architecture can learn the signature we
encoded. **They are not field accuracy.** They become field accuracy when
`ml/train.py` is re-run unchanged on real collar data. Until then, quote them
as what they are.

## Quick start

```bash
git clone <this repo> && cd smartherd
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate the dataset (~30 s)
python simulator/simulate_collar.py --out data/telemetry.npz --animals 60 --days 14

# 2. train, ablate, and benchmark against a classical baseline (~3 min, CPU)
python ml/train.py --data data/telemetry.npz --epochs 25

# 3. figures
python ml/make_figures.py

# 4. watch the whole pipeline run end to end, no hardware, no serial port
python simulator/proteus_bridge.py replay --minutes 2880
```

Everything it computes is written to `ml/artifacts/metrics.json`. Nothing in
that file is typed by hand, and the figure script reads only from it, so the
figures cannot disagree with the numbers.

## The Proteus bench

The collar is simulated twice: in Proteus VSM for the live demonstration, and
in Python for training. Both emit the same telemetry line, so nothing
downstream knows the difference.

See [`firmware/PROTEUS_SETUP.md`](firmware/PROTEUS_SETUP.md) for the
schematic, the component list and the demo script. The short version: an
Arduino UNO with an LM35 for temperature, a potentiometer standing in for the
PPG front end, COMPIM injecting NMEA, and a virtual terminal carrying the
telemetry out.

The moment worth showing a judge is sweeping the potentiometer up while the
animal stays still. Heart rate climbs with no movement to explain it, and
within two windows the classifier moves from `resting` to `illness_suspect`
and the risk engine returns the disease shortlist. That gesture is the entire
physio-spatial argument.

That guide also lists what the bench cannot do — the LM35 quantises to about
0.49 °C per ADC count, a potentiometer is not a PPG, and there is no power
model — because a judge will find those anyway.

## Repository layout

```
data/
  sadc_animal_diseases_2011.csv     78 diseases, signs, control, zoonotic
                                    risk, SADC 2011 burden, Botswana
                                    wildlife hosts
  indigenous_knowledge/
    badisa_indicators.md            herder signs mapped to sensor channels
    setswana_livestock_lexicon.csv  terms, with the gaps marked as gaps
firmware/
  proteus_collar.ino                sketch that runs inside Proteus
  PROTEUS_SETUP.md                  schematic, wiring, demo script, limits
simulator/
  simulate_collar.py                the disclosed generator
  proteus_bridge.py                 feed / read / replay
ml/
  dataset.py                        animal-level splits, no leakage
  model.py                          1D CNN-GRU
  train.py                          training, ablation, classical baseline
  risk_engine.py                    model output -> routed, explained alert
  make_figures.py
  artifacts/metrics.json            every number this project reports
docs/
  ARCHITECTURE.md
  DATA_ETHICS.md
demo/index.html                     the live demo page
```

## Roadmap

Nearest first:

1. **Validate the badisa indicators.** Two herders, different districts, not
   related. This is the cheapest and highest-value work left.
2. **Complete the Setswana disease lexicon with DVS.** The gaps in
   `setswana_livestock_lexicon.csv` are marked as gaps on purpose — for
   anthrax and rabies a wrong term is a safety problem, not a translation
   problem.
3. **Replace the synthetic study with a field trial**, however small, and
   re-run `ml/train.py` unchanged.
4. **Add a jaw or neck accelerometer** for rumination, the highest-value
   herder sign we currently cannot sense.
5. **Quantise to int8 and put the model on the collar.** At 33k parameters
   it is roughly 50 KB quantised, which is within reach of TFLite Micro on an
   ESP32-S3. On-collar inference means alerts survive a dead network.
6. **Integrate with BAITS** so a theft alert lands in the national
   traceability record rather than only on a phone.
7. **Herd-level aggregation.** With one collar the herd-cohesion channel
   degrades to distance-from-kraal. The design assumes several collars
   reporting to a shared centroid.

## Contributing

Talk to a herder, write down what they said in their words, and open a pull
request against `data/indigenous_knowledge/`. Record who and where. If an
indicator cannot be sensed with the hardware we have, add it to the gap list
anyway — the record is the point.

## Credits and sources

- SADC animal-disease data, 2011 annex, as transcribed in
  `data/sadc_animal_diseases_2011.csv`. Blank numeric fields are dashes in
  the source and do not distinguish zero from not-reported.
- Statistics Botswana, *Environment Statistics: Wildlife Digest 2014*, for
  wildlife population context. That digest reports populations and poaching,
  not disease occurrence; it is used as context only.
- WOAH Animal Diseases portal and the MSD Veterinary Manual for clinical
  summaries.
- SmartHerd.ai builds on the SmartHerd.ai collar work; the field power and geofence
  measurements from that project are **not** reproduced or claimed here.

SmartHerd.ai flags behaviour. It does not diagnose, and it does not replace a
veterinary officer or a herder. Every alert it raises says so.
