"""Bridge between the Proteus collar bench and the SmartHerd.ai pipeline.

Three modes:

  feed     generate NMEA sentences on a serial port for Proteus COMPIM
  read     read SmartHerd telemetry lines from a serial port, window them,
           classify, and print the routed alert
  replay   do both in one process with no serial hardware at all - the
           fallback if virtual serial ports will not cooperate

Usage:
    python simulator/proteus_bridge.py feed   --port /dev/pts/3 --minutes 240
    python simulator/proteus_bridge.py read   --port /dev/pts/4
    python simulator/proteus_bridge.py replay --minutes 240
"""

import argparse
import math
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, HERE)

from simulate_collar import (KRAAL_LAT, KRAAL_LON, FENCE_RADIUS_M, WINDOW,
                             SAMPLE_SECONDS, SAMPLES_PER_DAY, M_PER_DEG_LAT,
                             m_per_deg_lon, haversine_m, simulate_animal,
                             simulate_herd_centroid, plan_herd_drives,
                             apply_sensor_noise, window_label, CLASSES)


# ---------------------------------------------------------------------------
# NMEA generation
# ---------------------------------------------------------------------------

def _checksum(body: str) -> str:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


def to_nmea_rmc(lat, lon, t_seconds):
    """Build a $GPRMC sentence Proteus COMPIM can feed to the sketch."""
    def dm(v, is_lat):
        hemi = ("N" if v >= 0 else "S") if is_lat else ("E" if v >= 0 else "W")
        v = abs(v)
        d = int(v)
        m = (v - d) * 60.0
        width = 2 if is_lat else 3
        return f"{d:0{width}d}{m:07.4f}", hemi

    lat_s, lat_h = dm(lat, True)
    lon_s, lon_h = dm(lon, False)
    hh = int(t_seconds // 3600) % 24
    mm = int(t_seconds // 60) % 60
    ss = t_seconds % 60
    body = (f"GPRMC,{hh:02d}{mm:02d}{ss:05.2f},A,{lat_s},{lat_h},"
            f"{lon_s},{lon_h},0.0,0.0,010126,,")
    return f"${body}*{_checksum(body)}\r\n"


def build_track(minutes, seed=7):
    """One animal's ground truth track, reusing the training simulator so the
    bench and the training set cannot drift apart."""
    n = max(WINDOW + 2, int(minutes * 60 // SAMPLE_SECONDS))
    rng = np.random.default_rng(seed)
    drive = plan_herd_drives(n, rng)
    herd = simulate_herd_centroid(n, rng, drive=drive)
    lat, lon, temp, hr, labels, _ = simulate_animal(n, herd, rng, drive=drive)
    lat, lon, temp, hr = apply_sensor_noise(lat, lon, temp, hr, rng)
    return lat, lon, temp, hr, labels, herd


def cmd_feed(args):
    import serial  # pyserial, only needed in this mode
    lat, lon, temp, hr, labels, _ = build_track(args.minutes, args.seed)
    ser = serial.Serial(args.port, 9600, timeout=1)
    print(f"feeding {len(lat)} fixes to {args.port} at {args.rate}/s "
          f"(Ctrl-C to stop)")
    try:
        for i in range(len(lat)):
            ser.write(to_nmea_rmc(lat[i], lon[i],
                                  (i * SAMPLE_SECONDS) % 86400).encode())
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


# ---------------------------------------------------------------------------
# Feature assembly and inference
# ---------------------------------------------------------------------------

class WindowBuilder:
    """Turns a stream of telemetry samples into the same 10-channel window
    the model was trained on. The herd centroid is not observable from a
    single collar, so with one device it falls back to the kraal position -
    and the classifier's herd-cohesion channel degrades accordingly. With two
    or more collars reporting, pass the live centroid instead."""

    def __init__(self, window=WINDOW):
        self.window = window
        self.buf = deque(maxlen=window)
        self.speeds = deque(maxlen=3)
        self.prev = None
        self.prev_bearing = 0.0

    def push(self, lat, lon, temp_c, hr_bpm, t_index, herd_lat=None, herd_lon=None):
        if self.prev is None:
            step = 0.0
            bearing = 0.0
        else:
            step = float(haversine_m(self.prev[0], self.prev[1], lat, lon))
            bearing = math.atan2(lon - self.prev[1], lat - self.prev[0])
        turn = math.atan2(math.sin(bearing - self.prev_bearing),
                          math.cos(bearing - self.prev_bearing))
        self.prev = (lat, lon)
        self.prev_bearing = bearing

        speed = step / SAMPLE_SECONDS
        hlat = KRAAL_LAT if herd_lat is None else herd_lat
        hlon = KRAAL_LON if herd_lon is None else herd_lon
        herd_dist = float(haversine_m(lat, lon, hlat, hlon))
        kraal_dist = float(haversine_m(lat, lon, KRAAL_LAT, KRAAL_LON))
        fence_dist = FENCE_RADIUS_M - kraal_dist

        # activity_idx must be computed exactly as build_features() computes
        # it - a 3-sample moving average of speed squared. Using the raw
        # instantaneous value here silently shifts one input channel away from
        # what the model was trained on, and the classifier degrades without
        # ever raising an error.
        self.speeds.append(speed)
        activity = float(np.mean(np.square(self.speeds))) * 100.0

        hour = (t_index % SAMPLES_PER_DAY) * SAMPLE_SECONDS / 3600.0
        self.buf.append([speed, step, turn, herd_dist, fence_dist,
                         temp_c, hr_bpm, activity,
                         math.sin(2 * math.pi * hour / 24.0),
                         math.cos(2 * math.pi * hour / 24.0)])
        return fence_dist < 0.0

    def ready(self):
        return len(self.buf) == self.window

    def array(self):
        return np.array(self.buf, dtype=np.float32).T[None, ...]   # (1, C, T)


def load_model(path=None):
    import torch
    from model import CNNGRU
    path = path or os.path.join(ROOT, "ml", "artifacts", "smartherd_cnn_gru.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CNNGRU(in_channels=ck["in_channels"], n_classes=len(ck["classes"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck["mu"], ck["sd"], ck["classes"]


def classify(net, mu, sd, classes, X):
    import torch
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor((X - mu) / sd)), dim=1).numpy()[0]
    i = int(p.argmax())
    return classes[i], float(p[i]), p


def announce(alert):
    sev = alert["severity"].upper()
    if alert.get("suppressed"):
        print(f"  [{sev:8s}] {alert['prediction']} p={alert['probability']} "
              f"(below threshold, no alert sent)")
        return
    print(f"  [{sev:8s}] {alert['prediction']} p={alert['probability']} "
          f"-> {'/'.join(alert['channels'])}")
    print(f"    {alert['action']}")
    for d in alert.get("candidate_conditions", [])[:3]:
        z = " ZOONOTIC" if d["zoonotic"] else ""
        print(f"      - {d['disease']} (SADC 2011 outbreaks: "
              f"{d['sadc_2011_outbreaks']}){z}")


def run_stream(samples, herd=None, threshold=0.60, quiet=False):
    """samples: iterable of (lat, lon, temp_c, hr_bpm)."""
    from risk_engine import assess
    net, mu, sd, classes = load_model()
    wb = WindowBuilder()
    stride, n_alerts = 6, 0

    for i, (lat, lon, temp_c, hr_bpm) in enumerate(samples):
        hl = hln = None
        if herd is not None and i < len(herd):
            hl, hln = float(herd[i, 0]), float(herd[i, 1])
        breached = wb.push(lat, lon, temp_c, hr_bpm, i, hl, hln)
        if not wb.ready() or i % stride:
            continue
        pred, prob, _ = classify(net, mu, sd, classes, wb.array())
        hour = (i % SAMPLES_PER_DAY) * SAMPLE_SECONDS / 3600.0
        alert = assess(pred, prob, species="goat", animal_id="collar_sim_001",
                       fence_breached=breached, at_wildlife_interface=True,
                       night=(hour < 5 or hour > 21), threshold=threshold)
        if not quiet:
            print(f"t+{i*SAMPLE_SECONDS//60:5d} min  temp={temp_c:5.2f}C  "
                  f"hr={hr_bpm:5.1f}bpm")
            announce(alert)
        if not alert.get("suppressed") and alert["severity"] in ("high", "critical"):
            n_alerts += 1
    return n_alerts


def cmd_read(args):
    import serial
    ser = serial.Serial(args.port, 9600, timeout=2)
    print(f"reading SmartHerd telemetry from {args.port} (Ctrl-C to stop)")

    def gen():
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("SMARTHERD,"):
                continue
            p = line.split(",")
            if len(p) < 8 or p[7] != "1":
                continue
            try:
                yield float(p[3]), float(p[4]), float(p[5]), float(p[6])
            except ValueError:
                continue

    try:
        run_stream(gen(), threshold=args.threshold)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


def cmd_replay(args):
    lat, lon, temp, hr, labels, herd = build_track(args.minutes, args.seed)
    print(f"replaying {len(lat)} samples "
          f"({len(lat)*SAMPLE_SECONDS/3600:.1f} simulated hours)\n")
    n = run_stream(zip(lat, lon, temp, hr), herd=herd,
                   threshold=args.threshold, quiet=args.quiet)
    truth = {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))}
    print(f"\nground truth samples: {truth}")
    print(f"actionable alerts raised: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("feed", help="write NMEA to a serial port for COMPIM")
    f.add_argument("--port", required=True)
    f.add_argument("--minutes", type=int, default=240)
    f.add_argument("--rate", type=float, default=2.0, help="sentences per second")
    f.add_argument("--seed", type=int, default=7)
    f.set_defaults(func=cmd_feed)

    r = sub.add_parser("read", help="read telemetry from the Proteus terminal")
    r.add_argument("--port", required=True)
    r.add_argument("--threshold", type=float, default=0.60)
    r.set_defaults(func=cmd_read)

    p = sub.add_parser("replay", help="no serial port; simulate end to end")
    p.add_argument("--minutes", type=int, default=240)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_replay)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
