"""
SmartHerd.ai collar telemetry simulator.

Generates a labelled multi-channel time series for a herd of animals at a
Botswana cattlepost (moraka). There is no physical hardware in this build:
the collar is simulated here in software and, in parallel, in Proteus VSM
(see firmware/PROTEUS_SETUP.md). Both produce the same telemetry line format,
so the ML pipeline does not know or care which one fed it.

HONESTY NOTE (read this before quoting any number from this repo)
-----------------------------------------------------------------
Everything this file produces is SYNTHETIC. The generative parameters below
are grounded in three public sources:

  1. SADC Animal Disease data (data/sadc_animal_diseases_2011.csv) - the
     clinical sign strings ("Fever; ... lameness; production loss",
     "Sudden death; fever; ...") determine which channels an illness episode
     perturbs and in which direction.
  2. Published small-stock / cattle physiological baselines (rectal
     temperature, resting and active heart rate).
  3. Herder (badisa) field heuristics - see
     data/indigenous_knowledge/badisa_indicators.md - which determine the
     behavioural channels (herd-cohesion distance, time at rest, gait
     regularity).

Model scores computed on this data measure whether the architecture can
learn the signature we encoded. They are NOT field accuracy. They become
field accuracy only after the same protocol is re-run on real collar data.

Usage:
    python simulator/simulate_collar.py --out data/telemetry.npz --animals 60 --days 14
"""

import argparse
import json
import math
import os

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SAMPLE_SECONDS = 300          # 5-minute duty cycle (matches deployed collar)
SAMPLES_PER_DAY = 24 * 3600 // SAMPLE_SECONDS   # 288
WINDOW = 24                   # 24 samples = 2-hour decision window
STRIDE = 6                    # 30-minute hop between windows

# Cattlepost near Tlokweng, Botswana. Inside the national bounds
# (lat -27.0..-17.7, lon 19.9..29.4) used by the SmartHerd platform.
KRAAL_LAT, KRAAL_LON = -24.6580, 25.9760
FENCE_RADIUS_M = 1200.0       # permitted grazing radius (masimo boundary)

M_PER_DEG_LAT = 110_574.0

CLASSES = [
    "grazing",           # 0 normal - active foraging
    "resting",           # 1 normal - kraal / shade, ruminating
    "illness_suspect",   # 2 febrile + hypoactive + isolated from herd
    "theft_suspect",     # 3 sustained directed night movement, fence exit
    "predator_flight",   # 4 short high-speed scatter, HR spike
]

# Channel order is fixed and is depended on by ml/dataset.py and the ablation.
CHANNELS = [
    "speed_ms",          # 0  spatial
    "step_m",            # 1  spatial
    "turn_rad",          # 2  spatial
    "herd_dist_m",       # 3  spatial (social cohesion - badisa heuristic)
    "fence_dist_m",      # 4  spatial (signed: + inside, - outside)
    "temp_c",            # 5  physiological
    "hr_bpm",            # 6  physiological
    "activity_idx",      # 7  physiological/kinematic
    "hour_sin",          # 8  temporal
    "hour_cos",          # 9  temporal
]
MOVEMENT_ONLY_CHANNELS = [0, 1, 2, 3, 4, 8, 9]   # used for the ablation


def m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------
# Behaviour scheduling
# --------------------------------------------------------------------------

def diurnal_state(sample_index: int) -> str:
    """Default behaviour from time of day, following cattlepost routine:
    released from kraal at dawn, grazes, shade-rests through midday heat,
    grazes again, returns to kraal before dark."""
    hour = (sample_index % SAMPLES_PER_DAY) * SAMPLE_SECONDS / 3600.0
    if 6.0 <= hour < 11.0:
        return "grazing"
    if 15.0 <= hour < 18.5:
        return "grazing"
    return "resting"


def base_speed(state, rng):
    if state == "grazing":
        return abs(rng.normal(0.32, 0.14))
    return abs(rng.normal(0.025, 0.02))


def base_temp(sample_index, rng):
    """Small-stock rectal baseline ~38.6 C with a diurnal swing of ~0.35 C."""
    hour = (sample_index % SAMPLES_PER_DAY) * SAMPLE_SECONDS / 3600.0
    return 38.60 + 0.35 * math.sin(2 * math.pi * (hour - 4.0) / 24.0) + rng.normal(0, 0.08)


def base_hr(state, speed, rng):
    resting_hr = 76.0
    return resting_hr + 62.0 * min(speed, 1.2) + rng.normal(0, 4.0)


# --------------------------------------------------------------------------
# Episode injection
# --------------------------------------------------------------------------

def plan_episodes(n_samples, rng, drive=None):
    """Return a per-sample label array with abnormal episodes inserted.

    Rates are deliberately low and imbalanced, because that is the real
    operating condition: most of an animal's life is unremarkable, and a
    detector that only works on balanced data is useless on a farm.
    """
    labels = np.array([CLASSES.index(diurnal_state(i)) for i in range(n_samples)], dtype=np.int64)
    # mode drives the motion generator; it is not always the same thing as the
    # label, which is the point of the confounders below.
    mode = np.array(["rest" if labels[i] == CLASSES.index("resting") else "graze"
                     for i in range(n_samples)], dtype=object)
    episodes = []

    # --- Confounder A: midday shade rest during grazing hours -------------
    # In Botswana heat a perfectly healthy animal will stand in shade for
    # hours. Movement-wise this is indistinguishable from a sick animal.
    # Without this, "not moving when it should be grazing" is a free giveaway
    # and the physiological channels never have to earn their place.
    n_days = max(1, n_samples // SAMPLES_PER_DAY)
    for day in range(n_days):
        for _ in range(int(rng.integers(1, 4))):
            hour = rng.uniform(6.5, 18.0)
            start = day * SAMPLES_PER_DAY + int(hour * 3600 / SAMPLE_SECONDS)
            dur = int(rng.integers(6, 30))
            end = min(n_samples, start + dur)
            if start >= n_samples:
                continue
            labels[start:end] = CLASSES.index("resting")
            mode[start:end] = "rest"

    # --- Confounder B: legitimate herd drives ------------------------------
    if drive is not None:
        idx = np.where(drive)[0]
        labels[idx] = CLASSES.index("grazing")   # normal, farmer-initiated
        mode[idx] = "drive"

    # --- Illness episodes -------------------------------------------------
    # Onset is gradual (fever rises over ~6h then plateaus), duration 1-3 days.
    # Signs mapped from SADC clinical-sign strings: fever + lameness +
    # reduced grazing + separation from the herd.
    for _ in range(rng.integers(0, 3)):
        dur = int(rng.integers(SAMPLES_PER_DAY, 3 * SAMPLES_PER_DAY))
        if dur >= n_samples:
            continue
        start = int(rng.integers(0, n_samples - dur))
        labels[start:start + dur] = CLASSES.index("illness_suspect")
        mode[start:start + dur] = "ill"
        # A third of episodes are early/subclinical: little or no fever yet,
        # only a mild tachycardia and a slight loss of condition. These are
        # the ones worth catching and the ones the model will miss.
        episodes.append({"type": "illness_suspect", "start": start, "dur": dur,
                         "subclinical": bool(rng.random() < 0.33)})

    # --- Theft episodes ---------------------------------------------------
    # Rustling in Botswana is overwhelmingly nocturnal and vehicle- or
    # drive-assisted: sustained, directed, away from the kraal.
    for _ in range(rng.integers(0, 3)):
        dur = int(rng.integers(18, 60))          # 1.5 - 5 hours
        day = int(rng.integers(0, max(1, n_samples // SAMPLES_PER_DAY)))
        night_start = day * SAMPLES_PER_DAY + int(22.5 * 3600 / SAMPLE_SECONDS)
        start = night_start + int(rng.integers(0, 24))
        if start + dur >= n_samples:
            continue
        labels[start:start + dur] = CLASSES.index("theft_suspect")
        mode[start:start + dur] = "theft"
        episodes.append({"type": "theft_suspect", "start": start, "dur": dur})

    # --- Predator / intruder flight events --------------------------------
    for _ in range(rng.integers(0, 5)):
        dur = int(rng.integers(2, 6))            # 10 - 30 minutes
        start = int(rng.integers(0, n_samples - dur))
        if labels[start] in (CLASSES.index("theft_suspect"),):
            continue
        labels[start:start + dur] = CLASSES.index("predator_flight")
        mode[start:start + dur] = "flight"
        episodes.append({"type": "predator_flight", "start": start, "dur": dur})

    return labels, mode, episodes


def simulate_animal(n_samples, herd_track, rng, drive=None):
    """Simulate one animal. herd_track is the (n,2) lat/lon of the herd
    centroid, so that social isolation is measurable."""
    labels, mode, episodes = plan_episodes(n_samples, rng, drive=drive)

    lat = np.zeros(n_samples)
    lon = np.zeros(n_samples)
    temp = np.zeros(n_samples)
    hr = np.zeros(n_samples)

    lat[0] = KRAAL_LAT + rng.normal(0, 30) / M_PER_DEG_LAT
    lon[0] = KRAAL_LON + rng.normal(0, 30) / m_per_deg_lon(KRAAL_LAT)
    heading = rng.uniform(0, 2 * math.pi)

    fever_peak = rng.uniform(1.3, 2.6)
    theft_heading = rng.uniform(0, 2 * math.pi)
    # How strongly this animal lags the herd when sick. Some sick animals
    # stay with the herd; assuming they all separate would be wishful.
    lag_tendency = rng.uniform(0.0, 1.0)

    for i in range(n_samples):
        md = mode[i]

        if md == "graze":
            spd = base_speed("grazing", rng)
            heading += rng.normal(0, 0.9)
            pull = 0.12
        elif md == "rest":
            spd = base_speed("resting", rng)
            heading += rng.normal(0, 1.4)
            pull = 0.12
        elif md == "drive":
            # Whole herd moving together, daylight, under human control.
            spd = abs(rng.normal(1.9, 0.45))
            heading += rng.normal(0, 0.15)
            pull = 0.45                      # tight cohesion is the tell
        elif md == "ill":
            sched = diurnal_state(i)
            spd = base_speed(sched, rng) * rng.uniform(0.05, 0.55)
            heading += rng.normal(0, 1.2)
            # A sick animal lags the herd, it does not emigrate. Letting it
            # drift freely for two days would hand the classifier a
            # monotonically growing distance-to-herd signal that no real sick
            # animal produces, and would make the movement-only ablation look
            # far better than it deserves to.
            pull = 0.12 - 0.04 * lag_tendency
        elif md == "theft":
            spd = abs(rng.normal(2.6, 0.8))
            heading = theft_heading + rng.normal(0, 0.10)
            pull = 0.0                       # separated from the herd
        else:  # flight
            spd = abs(rng.normal(4.2, 1.0))
            heading += rng.normal(0, 1.1)
            pull = 0.05

        if i > 0:
            dist = spd * SAMPLE_SECONDS
            dlat = (dist * math.cos(heading)) / M_PER_DEG_LAT
            dlon = (dist * math.sin(heading)) / m_per_deg_lon(KRAAL_LAT)
            lat[i] = lat[i - 1] + dlat + pull * (herd_track[i, 0] - lat[i - 1])
            lon[i] = lon[i - 1] + dlon + pull * (herd_track[i, 1] - lon[i - 1])

        # ---- physiology -------------------------------------------------
        t = base_temp(i, rng)
        h = base_hr("resting" if md == "ill" else md, spd, rng)

        if md == "ill":
            ep = next((e for e in episodes
                       if e["type"] == "illness_suspect"
                       and e["start"] <= i < e["start"] + e["dur"]), None)
            if ep:
                ramp = min(1.0, (i - ep["start"]) / 72.0)
                if ep.get("subclinical"):
                    t += 0.35 * fever_peak * ramp
                    h += 12.0 * ramp
                else:
                    t += fever_peak * ramp
                    h += 28.0 * ramp
        elif md == "flight":
            h += rng.uniform(45, 75)
        elif md == "theft":
            h += rng.uniform(10, 25)
        elif md == "drive":
            h += rng.uniform(8, 20)          # exertion, but no fever

        temp[i] = t
        hr[i] = h

    return lat, lon, temp, hr, labels, episodes


def plan_herd_drives(n_samples, rng):
    """Legitimate herd movements: being walked to the dip tank, to a new
    grazing camp, or home ahead of a storm. These look like theft in raw
    speed terms - fast, sustained, directed, away from the kraal - and are
    the single most important confounder in this problem. A detector that
    has never seen one will cry wolf every dipping day.
    """
    drive = np.zeros(n_samples, dtype=bool)
    n_days = max(1, n_samples // SAMPLES_PER_DAY)
    for day in range(n_days):
        if rng.random() < 0.35:                     # roughly twice a week
            hour = rng.uniform(7.0, 16.0)           # always in daylight
            start = day * SAMPLES_PER_DAY + int(hour * 3600 / SAMPLE_SECONDS)
            dur = int(rng.integers(12, 40))         # 1 - 3.5 hours
            if start + dur < n_samples:
                drive[start:start + dur] = True
    return drive


def simulate_herd_centroid(n_samples, rng, drive=None):
    lat = np.zeros(n_samples)
    lon = np.zeros(n_samples)
    lat[0], lon[0] = KRAAL_LAT, KRAAL_LON
    heading = rng.uniform(0, 2 * math.pi)
    drive_heading = rng.uniform(0, 2 * math.pi)
    for i in range(1, n_samples):
        driving = drive is not None and drive[i]
        if driving:
            spd = abs(rng.normal(1.9, 0.4))
            heading = drive_heading + rng.normal(0, 0.12)
        else:
            state = diurnal_state(i)
            spd = 0.28 if state == "grazing" else 0.02
            heading += rng.normal(0, 0.35)
        dist = spd * SAMPLE_SECONDS
        lat[i] = lat[i - 1] + (dist * math.cos(heading)) / M_PER_DEG_LAT
        lon[i] = lon[i - 1] + (dist * math.sin(heading)) / m_per_deg_lon(KRAAL_LAT)
        if not driving:
            lat[i] += 0.05 * (KRAAL_LAT - lat[i])
            lon[i] += 0.05 * (KRAAL_LON - lon[i])
        else:
            drive_heading += rng.normal(0, 0.02)
    return np.stack([lat, lon], axis=1)


# --------------------------------------------------------------------------
# Sensor noise (this is where Proteus and the real world agree)
# --------------------------------------------------------------------------

def apply_sensor_noise(lat, lon, temp, hr, rng):
    n = len(lat)
    # GNSS error is not white: consumer modules show strong lag-1 correlation.
    sigma_m, rho = 3.2, 0.6
    e_lat = np.zeros(n); e_lon = np.zeros(n)
    for i in range(1, n):
        e_lat[i] = rho * e_lat[i - 1] + rng.normal(0, sigma_m * math.sqrt(1 - rho ** 2))
        e_lon[i] = rho * e_lon[i - 1] + rng.normal(0, sigma_m * math.sqrt(1 - rho ** 2))
    lat = lat + e_lat / M_PER_DEG_LAT
    lon = lon + e_lon / m_per_deg_lon(KRAAL_LAT)

    # LM35 in the Proteus bench + 10-bit ADC quantisation
    temp = np.round(temp + rng.normal(0, 0.22, n), 2)
    # PPG is the least reliable channel on a hairy animal: noise + dropouts
    hr = hr + rng.normal(0, 5.5, n)
    dropout = rng.random(n) < 0.04
    hr[dropout] = np.nan
    # forward-fill dropouts the way the firmware would
    for i in range(1, n):
        if np.isnan(hr[i]):
            hr[i] = hr[i - 1]
    hr = np.nan_to_num(hr, nan=76.0)
    return lat, lon, temp, hr


# --------------------------------------------------------------------------
# Feature extraction (identical code path is used on live telemetry)
# --------------------------------------------------------------------------

def build_features(lat, lon, temp, hr, herd_track):
    n = len(lat)
    step = np.zeros(n)
    step[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    speed = step / SAMPLE_SECONDS

    bearing = np.zeros(n)
    bearing[1:] = np.arctan2(lon[1:] - lon[:-1], lat[1:] - lat[:-1])
    turn = np.zeros(n)
    turn[1:] = np.arctan2(np.sin(bearing[1:] - bearing[:-1]),
                          np.cos(bearing[1:] - bearing[:-1]))

    herd_dist = haversine_m(lat, lon, herd_track[:, 0], herd_track[:, 1])
    kraal_dist = haversine_m(lat, lon, KRAAL_LAT, KRAAL_LON)
    fence_dist = FENCE_RADIUS_M - kraal_dist        # + inside, - outside

    # Activity index: rolling kinematic energy proxy, what an accelerometer
    # would report. Computed from position so the software and Proteus
    # benches stay comparable.
    activity = np.convolve(speed ** 2, np.ones(3) / 3.0, mode="same") * 100.0

    idx = np.arange(n)
    hour = (idx % SAMPLES_PER_DAY) * SAMPLE_SECONDS / 3600.0
    hour_sin = np.sin(2 * math.pi * hour / 24.0)
    hour_cos = np.cos(2 * math.pi * hour / 24.0)

    return np.stack([speed, step, turn, herd_dist, fence_dist,
                     temp, hr, activity, hour_sin, hour_cos], axis=0)


def window_label(lab_seg, window=WINDOW):
    """Collapse a window's per-sample labels into one window label.

    Used by both the training set builder and the live bridge, so a window is
    never labelled one way during training and another way at inference.
    """
    counts = np.bincount(np.asarray(lab_seg, dtype=np.int64), minlength=len(CLASSES))
    # Per-class occupancy thresholds. A majority rule would erase acute
    # events: a 20-minute predator scatter is only 8% of a 2-hour window but
    # is exactly what the farmer needs to hear about. Thresholds are ordered
    # by operational urgency, not by frequency.
    i_theft = CLASSES.index("theft_suspect")
    i_flight = CLASSES.index("predator_flight")
    i_ill = CLASSES.index("illness_suspect")
    if counts[i_theft] >= window * 0.25:
        return i_theft
    if counts[i_flight] >= window * 0.08:
        return i_flight
    if counts[i_ill] >= window * 0.50:
        return i_ill
    return int(np.argmax(counts[:2]))


def window_series(feat, labels, window=WINDOW, stride=STRIDE):
    X, y = [], []
    n = feat.shape[1]
    for s in range(0, n - window + 1, stride):
        X.append(feat[:, s:s + window])
        y.append(window_label(labels[s:s + window], window))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/telemetry.npz")
    ap.add_argument("--animals", type=int, default=60)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_samples = args.days * SAMPLES_PER_DAY
    drive = plan_herd_drives(n_samples, rng)
    herd_track = simulate_herd_centroid(n_samples, rng, drive=drive)

    Xs, ys, animal_ids = [], [], []
    for a in range(args.animals):
        lat, lon, temp, hr, labels, _ = simulate_animal(n_samples, herd_track, rng, drive=drive)
        lat, lon, temp, hr = apply_sensor_noise(lat, lon, temp, hr, rng)
        feat = build_features(lat, lon, temp, hr, herd_track)
        X, y = window_series(feat, labels)
        Xs.append(X); ys.append(y)
        animal_ids.append(np.full(len(y), a, dtype=np.int64))

    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(animal_ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, animal=g,
                        channels=np.array(CHANNELS), classes=np.array(CLASSES))

    dist = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
    meta = {
        "windows": int(len(y)), "animals": args.animals, "days": args.days,
        "window_samples": WINDOW, "stride": STRIDE,
        "sample_seconds": SAMPLE_SECONDS, "seed": args.seed,
        "channels": CHANNELS, "classes": CLASSES,
        "class_distribution": dist,
        "synthetic": True,
        "confounders": ["midday shade rest during grazing hours",
                        "legitimate herd drives (dip tank / camp move)",
                        "subclinical illness episodes with little or no fever",
                        "variable herd-lag tendency in sick animals"],
        "herd_drive_samples": int(drive.sum()),
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "telemetry_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
