# Building the SmartHerd.ai collar bench in Proteus

This build has no physical hardware. The collar is simulated twice: once in
Proteus VSM, which is what you demonstrate live, and once in pure Python
(`simulator/simulate_collar.py`), which is what generates the training set.
Both emit the same telemetry line, so nothing downstream knows the
difference.

Expect this to take about 30 minutes the first time.

## Why an Arduino UNO and not an ESP32

Proteus has no native ESP32 part. Rather than fight that, the bench uses an
ATmega328P, which Proteus simulates cycle-accurately. What the bench is
testing is the sensor chain, the parsing, the telemetry format and the alert
path. None of that needs an ESP32. The parts that do — TLS, the modem stack,
deep sleep — are not simulated here, and the README says so rather than
implying a fidelity we do not have.

## Component list

Search these exact names in the Proteus device library:

| Part | Library name | Role |
|---|---|---|
| Microcontroller | `ARDUINO UNO R3` (Arduino library) or `ATMEGA328P` | runs `proteus_collar.ino` |
| Temperature sensor | `LM35` | body temperature proxy, A0 |
| Heart-rate proxy | `POT-HG` | potentiometer on A1, sweep it by hand to inject tachycardia |
| GPS source | `COMPIM` | injects NMEA sentences from a host serial port |
| Telemetry output | `VIRTUAL TERMINAL` | the line the bridge reads |
| Breach trigger | `SW-SPDT` | forces a position outside the fence |
| Indicator | `LED-RED` + `RES` (220R) | heartbeat on D13 |
| Supply | `POWER` / `GROUND` terminals | 5 V rail |

## Wiring

```
LM35        VOUT  -> A0            (VCC 5V, GND to ground)
POT-HG      wiper -> A1            (ends to 5V and GND)
SW-SPDT     common-> D2            (other pole to GND; D2 uses INPUT_PULLUP)
LED-RED     anode -> 220R -> D13   (cathode to GND)
COMPIM      TXD   -> D8            (sketch RX, SoftwareSerial)
COMPIM      RXD   -> D9            (sketch TX, unused but wire it)
ATMEGA328P  TXD(D1) -> VIRTUAL TERMINAL RXD
ATMEGA328P  RXD(D0) -> VIRTUAL TERMINAL TXD
```

Set both COMPIM and the virtual terminal to **9600 baud, 8 data bits, no
parity, 1 stop bit**. Mismatched baud is the single most common reason the
terminal shows nothing but garbage.

## Loading the sketch

1. Compile `firmware/proteus_collar.ino` in the Arduino IDE with board
   "Arduino Uno". Turn on verbose output during compilation in Preferences
   so the IDE prints the path to the generated `.hex`.
2. In Proteus, double-click the UNO, and set **Program File** to that `.hex`.
3. Set the **CKSEL** / clock frequency property to 16 MHz.

## Feeding it GPS

`COMPIM` needs a serial port with NMEA on it. Two options.

**Option A, virtual serial pair (what we use).** Install `com0com` on
Windows or use `socat` on Linux to create a linked pair, point COMPIM at one
end and the feeder at the other:

```bash
# Linux
socat -d -d pty,raw,echo=0 pty,raw,echo=0
# note the two /dev/pts/N paths it prints
python3 simulator/proteus_bridge.py feed --port /dev/pts/3 --minutes 240
```

**Option B, file replay.** If you cannot get a virtual pair working in the
time you have, skip COMPIM entirely. Run the bridge in `replay` mode, which
generates the same NMEA stream and writes telemetry straight to the pipeline,
and demonstrate Proteus separately for the sensor chain. This is the safe
fallback and it is not cheating — say out loud which one you are showing.

## Running the demo

1. Start the NMEA feeder (or the replay).
2. Press play in Proteus. The D13 LED should blink and the virtual terminal
   should start printing `SMARTHERD,...` lines.
3. Start the bridge in `read` mode; it windows the telemetry, runs the
   CNN-GRU and prints the routed alert.
4. Sweep the POT-HG upward while leaving the animal stationary. Heart rate
   climbs with no corresponding movement, and within a couple of windows the
   classifier moves from `resting` to `illness_suspect` and the risk engine
   returns the SADC shortlist. **This is the moment to show a judge** — it is
   the whole physio-spatial argument in one gesture.
5. Flip the SW-SPDT. Position jumps outside the fence and the deterministic
   breach alert fires regardless of what the model thinks. Point out that the
   model can raise the priority of that alert but can never cancel it.

## Known limits of this bench

State these rather than let a judge find them:

- The LM35 on a 5 V reference and a 10-bit ADC resolves about 0.49 °C per
  count. Fever ramps are quantised. The sketch oversamples 16× to recover
  roughly two bits; a real collar uses a digital sensor.
- The potentiometer is not a PPG. It proves the signal path, not that a
  heart rate can be extracted from an animal with hair on it.
- There is no power model here. The 21.8 mA / 91 h figure from the SmartHerd
  hardware work is a measurement from real hardware and is not reproduced or
  claimed by this bench.
- Proteus timing is compressed 150:1 against the real 300 s duty cycle.
