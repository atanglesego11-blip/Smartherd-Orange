# Badisa indicators: herder knowledge as model features

A *modisa* (herder; plural *badisa*) who has walked with the same herd for
years can tell a sick animal from a healthy one at fifty metres, before any
lesion is visible. That skill is real, it is diagnostic, and it is
disappearing — it lives in the heads of older herders, it is transmitted by
walking alongside them, and fewer young people are doing that walk.

SmartHerd.ai's design position is that this knowledge is **not decoration on top of
a machine-learning model. It is the feature set.** Every channel the
classifier reads was chosen because a herder already uses it. The model does
not know anything a good *modisa* doesn't; what it has is patience, and the
ability to watch two hundred animals at three in the morning.

This file is the mapping. It is also a live document: the "status" column
says which entries have been validated with practising badisa and which are
still our reading of the literature. **Entries marked `unverified` must not
be presented as indigenous knowledge until a herder has confirmed them.**

## The mapping

| # | Herder observation | What it means in practice | Sensor channel(s) in SmartHerd.ai | Status |
|---|---|---|---|---|
| 1 | The animal *goes behind* — lags the herd, arrives last at water | Earliest general sign of illness or injury; precedes visible lesions | `herd_dist_m` (distance to herd centroid), trend over hours | unverified |
| 2 | Stands apart from the others, away from the group at rest | Social withdrawal; strong non-specific illness sign | `herd_dist_m` during rest hours | unverified |
| 3 | Does not graze when the others graze | Loss of appetite. **The hard case:** a healthy animal standing in shade at midday looks identical | `speed_ms`, `activity_idx` during grazing hours — only separable when read together with `temp_c` and `hr_bpm` | unverified |
| 4 | Walks badly, favours a foot, short hesitant steps | Lameness. Central to foot-and-mouth and footrot detection | `step_m` distribution, `turn_rad` variance within a window | unverified |
| 5 | Stops chewing the cud when resting | Rumination ceases early in many febrile conditions | **Not yet sensed.** Needs a jaw/neck accelerometer. Listed here as a known gap | gap |
| 6 | Feels hot at the ear or horn base | Fever, judged by hand | `temp_c` | unverified |
| 7 | Breathing fast while standing still | Fever or respiratory disease; also tachycardia | `hr_bpm` decoupled from `speed_ms` — high heart rate with no movement | unverified |
| 8 | The herd scatters suddenly and comes back together | Predator, stray dogs, or people in the grazing area | `speed_ms` burst + `turn_rad` variance + `herd_dist_m` collapse and recovery | unverified |
| 9 | One or two animals leave in a straight line at night | Rustling. Animals do not walk in straight lines at night of their own accord | `speed_ms` sustained, `turn_rad` variance near zero, `hour_sin`/`hour_cos`, `fence_dist_m` | unverified |
| 10 | The whole herd moving fast in daylight is normal — it is being driven | Dip tank day, moving camp, coming home before a storm | Same kinematics as #9, separated by `herd_dist_m` staying small and by time of day | unverified |

Indicator 10 is why this file exists. A system built only from indicator 9
sends the farmer a theft alarm every dipping day, the farmer stops reading
the alarms, and the one real theft goes unanswered. The knowledge that
distinguishes 9 from 10 is not in any dataset we could download. It came
from asking.

## The gap list is the honest part

Indicator 5 (rumination) is the single most-cited herder sign we cannot
currently measure. We are listing it rather than quietly dropping it,
because a system that only implements the knowledge it finds convenient is
not preserving anything.

Also not yet implemented, and named here so they are not lost:

- Coat and skin condition, judged visually
- Colour and consistency of dung
- Whether the animal drinks, and how much
- Reading a specific animal's normal temperament, which is per-animal and
  is what makes a herder's judgement better than a threshold

## How to add to this file

Talk to a herder. Write down what they said, in their words, in the table.
Set status to `verified` only when you have spoken to at least two badisa
who are not related and who herd in different areas. Record who and where in
`contributions.md`. If an indicator cannot be sensed with the hardware we
have, add it to the gap list anyway — the record is the point.

## Attribution and ownership

Knowledge recorded here belongs to the communities it came from, not to this
project. It is published under the repository licence so it cannot be
enclosed, and contributors are named unless they ask not to be. No
individual animal, farmer, or cattlepost location is published without the
owner's consent — see `docs/DATA_ETHICS.md`.
