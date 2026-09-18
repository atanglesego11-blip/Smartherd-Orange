/*
 * SmartHerd.ai - simulated collar firmware for the Proteus VSM bench.
 *
 * There is no physical collar in this build. This sketch runs inside
 * Proteus on an ATmega328P (Arduino UNO) and produces exactly the same
 * telemetry line that the real SmartHerd collar produces, so the ML
 * pipeline downstream cannot tell the two apart. See PROTEUS_SETUP.md
 * for the schematic and the component list.
 *
 * Why an UNO and not an ESP32: Proteus has no native ESP32 model. The
 * UNO is simulated cycle-accurately, which is what we actually want from
 * a bench - deterministic, repeatable sensor timing. The parts of the
 * real firmware that an UNO cannot host (TLS, the modem stack, deep
 * sleep) are not what this bench is testing.
 *
 * Wiring (see PROTEUS_SETUP.md):
 *   A0  LM35 analog temperature sensor    -> body temperature proxy
 *   A1  POT-HG potentiometer              -> PPG / heart-rate proxy
 *   D2  SW-SPDT switch                    -> forces a fence-breach test case
 *   D8/D9 SoftwareSerial to COMPIM        -> NMEA GPS sentences in
 *   D13 LED                               -> heartbeat, proves the loop is alive
 *   Hardware UART (D0/D1) -> VIRTUAL TERMINAL -> telemetry out
 *
 * Telemetry line format (one per duty cycle), consumed by
 * simulator/proteus_bridge.py:
 *
 *   SMARTHERD,<deviceId>,<utc>,<lat>,<lon>,<temp_c>,<hr_bpm>,<fix>
 *
 * Licence: MIT. See LICENSE at the repository root.
 */

#include <SoftwareSerial.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const char DEVICE_ID[] = "sh_collar_sim_001";

const uint8_t PIN_TEMP     = A0;
const uint8_t PIN_PPG      = A1;
const uint8_t PIN_BREACH   = 2;
const uint8_t PIN_LED      = 13;
const uint8_t PIN_GPS_RX   = 8;
const uint8_t PIN_GPS_TX   = 9;

// Real collar duty cycle is 300 s. That is unbearable on a bench, so the
// simulation clock is compressed. Keep SAMPLE_PERIOD_MS in sync with the
// --speedup flag on proteus_bridge.py or the timestamps will drift.
const unsigned long SAMPLE_PERIOD_MS = 2000UL;   // 2 s bench == 300 s field

SoftwareSerial gpsSerial(PIN_GPS_RX, PIN_GPS_TX);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
char   nmeaBuf[96];
uint8_t nmeaLen = 0;

float  lastLat = 0.0f, lastLon = 0.0f;
char   lastUtc[11] = "000000.00";
bool   haveFix = false;

unsigned long lastSample = 0;

// ---------------------------------------------------------------------------
// Sensors
// ---------------------------------------------------------------------------

/* LM35 outputs 10 mV per degree C. With the default 5 V reference and a
 * 10-bit ADC that is 4.887 mV per count, so roughly 0.49 C per count. That
 * quantisation is coarse for a 0.3 C fever ramp, which is exactly the point
 * of simulating it: the bench shows you the resolution problem before you
 * have soldered anything. On the real collar this is a digital sensor. */
float readTemperatureC() {
  uint16_t acc = 0;
  for (uint8_t i = 0; i < 16; i++) {          // oversample to claw back ~2 bits
    acc += analogRead(PIN_TEMP);
    delay(2);
  }
  float counts = acc / 16.0f;
  return (counts * 5.0f / 1024.0f) * 100.0f;
}

/* The potentiometer stands in for a PPG front end. Sweep it during a run to
 * inject a tachycardia episode by hand and watch the classifier react. */
float readHeartRateBpm() {
  uint16_t raw = analogRead(PIN_PPG);
  return 40.0f + (raw / 1023.0f) * 160.0f;    // 40 - 200 bpm
}

// ---------------------------------------------------------------------------
// GPS - minimal $GPRMC parser
// ---------------------------------------------------------------------------

static float nmeaToDecimal(const char *field, char hemi) {
  // ddmm.mmmm or dddmm.mmmm
  const char *dot = strchr(field, '.');
  if (!dot) return 0.0f;
  uint8_t degDigits = (uint8_t)(dot - field) - 2;
  char degBuf[4] = {0};
  memcpy(degBuf, field, degDigits);
  float deg = atof(degBuf);
  float min = atof(field + degDigits);
  float dec = deg + min / 60.0f;
  if (hemi == 'S' || hemi == 'W') dec = -dec;
  return dec;
}

static void parseRMC(char *s) {
  // $GPRMC,utc,status,lat,N,lon,E,speed,course,date,...
  char *f[13] = {0};
  uint8_t n = 0;
  for (char *p = strtok(s, ","); p && n < 13; p = strtok(NULL, ",")) f[n++] = p;
  if (n < 7) return;
  if (f[2] == NULL || f[2][0] != 'A') { haveFix = false; return; }

  strncpy(lastUtc, f[1], sizeof(lastUtc) - 1);
  lastLat = nmeaToDecimal(f[3], f[4][0]);
  lastLon = nmeaToDecimal(f[5], f[6][0]);
  haveFix = true;
}

static void pumpGps() {
  while (gpsSerial.available()) {
    char c = gpsSerial.read();
    if (c == '\n' || c == '\r') {
      if (nmeaLen > 6) {
        nmeaBuf[nmeaLen] = '\0';
        if (strncmp(nmeaBuf, "$GPRMC", 6) == 0) parseRMC(nmeaBuf);
      }
      nmeaLen = 0;
    } else if (nmeaLen < sizeof(nmeaBuf) - 1) {
      nmeaBuf[nmeaLen++] = c;
    }
  }
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(9600);
  gpsSerial.begin(9600);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BREACH, INPUT_PULLUP);
  Serial.println(F("# SmartHerd.ai Proteus collar bench"));
  Serial.println(F("# SMARTHERD,device,utc,lat,lon,temp_c,hr_bpm,fix"));
}

void loop() {
  pumpGps();

  if (millis() - lastSample < SAMPLE_PERIOD_MS) return;
  lastSample = millis();

  digitalWrite(PIN_LED, !digitalRead(PIN_LED));

  float tempC = readTemperatureC();
  float hr    = readHeartRateBpm();

  // The manual breach switch nudges the reported position outside the fence.
  // It exists so the deterministic fence path can be tested without waiting
  // for the NMEA feed to reach a boundary.
  float lat = lastLat, lon = lastLon;
  if (digitalRead(PIN_BREACH) == LOW) {
    lat += 0.0300f;          // ~3.3 km north, well outside any fence
  }

  Serial.print(F("SMARTHERD,"));
  Serial.print(DEVICE_ID);        Serial.print(',');
  Serial.print(lastUtc);          Serial.print(',');
  Serial.print(lat, 6);           Serial.print(',');
  Serial.print(lon, 6);           Serial.print(',');
  Serial.print(tempC, 2);         Serial.print(',');
  Serial.print(hr, 1);            Serial.print(',');
  Serial.println(haveFix ? '1' : '0');
}
