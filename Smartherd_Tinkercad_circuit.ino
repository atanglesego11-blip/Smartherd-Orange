// ---------------------------------------------------------------- identity
const char DEVICE_ID[]   = "collar_001";
const char FARM_ID[]     = "cjafRzyuaiLKb3RQDnnj";   // confirm against farms table
const char ANIMAL_ID[]   = "goat_001";
const char ANIMAL_NAME[] = "Goat 001";
const char ANIMAL_TYPE[] = "goat";

// Shown in the printed HTTP frame only. The real key lives in the bridge,
// never in a sketch you paste into a public Tinkercad project.
const char SUPABASE_HOST[] = "https://ipuheybcnznqcsxobudg.supabase.co/functions/v1/ingest-gps";
const char INGEST_PATH[]   = "/functions/v1/ingest-gps";

// ---------------------------------------------------------------- hardware
const int TEMP_PIN  = A0;
const int PULSE_PIN = A1;
const int LAT_PIN   = A2;
const int LON_PIN   = A3;

const int GREEN_LED = 7;
const int RED_LED   = 8;
const int BUZZER    = 9;

// ---------------------------------------------------------------- thresholds
const float TEMP_MIN = 35.0;
const float TEMP_MAX = 40.0;
const int   PULSE_MIN = 40;
const int   PULSE_MAX = 140;

// Circular bench fence, centre near the test farm outside Gaborone.
// Coordinates are held as integer 1e-5 degrees throughout: an AVR float
// carries only ~7 significant digits, which is ~4 m of error at this
// latitude — enough to flip a fence decision on its own.
const long FENCE_LAT_E5 = -2460456L;   // -24.60456
const long FENCE_LON_E5 =  2592176L;   //  25.92176
const long FENCE_RADIUS_M = 200;

// Pot sweep covers roughly 400 m of lat and 400 m of lon around that centre,
// so turning a knob walks the animal in and out of the fence.
const long LAT_MIN_E5 = -2460700L, LAT_MAX_E5 = -2460300L;
const long LON_MIN_E5 =  2591900L, LON_MAX_E5 =  2592300L;

// 3 s is bench-only. A real run at this rate writes ~28,800 rows a day.
const unsigned long UPLOAD_INTERVAL_MS = 3000;
unsigned long lastUpdate = 0;
unsigned long seq = 0;

String lastPayload = "";

// ---------------------------------------------------------------- helpers

// Exact decimal from 1e-5 degrees. Avoids float rounding in the payload.
String degStr(long e5) {
  bool neg = e5 < 0;
  unsigned long a = neg ? (unsigned long)(-e5) : (unsigned long)e5;
  char buf[16];
  sprintf(buf, "%s%lu.%05lu", neg ? "-" : "", a / 100000UL, a % 100000UL);
  return String(buf);
}

// Same NaN guard as the real firmware. String(NAN) emits "nan", which is not
// valid JSON, and req.json() then throws before the insert is ever attempted.
String jsonNumber(float v, byte dp) {
  if (isnan(v) || isinf(v)) return F("null");
  return String(v, dp);
}

// Equirectangular distance, integer degrees in, metres out. At 1e-5 degrees
// one unit is 1.1132 m of latitude and 1.0122 m of longitude at -24.6.
long fenceDistanceM(long latE5, long lonE5) {
  float dLat = (float)(latE5 - FENCE_LAT_E5) * 1.1132;
  float dLon = (float)(lonE5 - FENCE_LON_E5) * 1.0122;
  return (long)(sqrt(dLat * dLat + dLon * dLon) + 0.5);
}

String buildPayload(long latE5, long lonE5, float tempC, int pulse,
                    bool insideFence, bool healthAlert) {
  String j = F("{");
  // seq keeps two identical fixes from a stationary animal distinguishable,
  // which is what the bridge's duplicate filter keys on.
  j += F("\"seq\":"); j += String(seq); j += F(",");
  j += F("\"deviceId\":\"");   j += DEVICE_ID;   j += F("\",");
  j += F("\"farmId\":\"");     j += FARM_ID;     j += F("\",");
  j += F("\"animalId\":\"");   j += ANIMAL_ID;   j += F("\",");
  j += F("\"animalName\":\""); j += ANIMAL_NAME; j += F("\",");
  j += F("\"animalType\":\""); j += ANIMAL_TYPE; j += F("\",");
  j += F("\"lat\":");  j += degStr(latE5); j += F(",");
  j += F("\"lon\":");  j += degStr(lonE5); j += F(",");
  j += F("\"latDir\":\""); j += (latE5 < 0 ? "S" : "N"); j += F("\",");
  j += F("\"lonDir\":\""); j += (lonE5 < 0 ? "W" : "E"); j += F("\",");
  j += F("\"alt\":");    j += jsonNumber(970.0, 1); j += F(",");
  j += F("\"speed\":");  j += jsonNumber(0.0, 2);   j += F(",");
  j += F("\"course\":"); j += jsonNumber(0.0, 2);   j += F(",");
  j += F("\"sats\":9,");
  j += F("\"insideFence\":"); j += (insideFence ? F("true") : F("false")); j += F(",");
  j += F("\"fenceVersion\":1,");
  // Physio channels. The current Edge Function ignores unknown keys, so these
  // ride along harmlessly until telemetry has columns for them.
  j += F("\"tempC\":");     j += jsonNumber(tempC, 1); j += F(",");
  j += F("\"pulseBpm\":");  j += String(pulse);        j += F(",");
  j += F("\"healthAlert\":"); j += (healthAlert ? F("true") : F("false"));
  j += F("}");
  return j;
}

// The request the ESP32 would put on the wire. Printed so the simulation
// demonstrates the protocol; the bridge is what actually sends it.
void printHttpFrame(const String& body) {
  Serial.println();
  Serial.print(F("POST ")); Serial.print(INGEST_PATH); Serial.println(F(" HTTP/1.1"));
  Serial.print(F("Host: ")); Serial.println(SUPABASE_HOST);
  Serial.println(F("Content-Type: application/json"));
  Serial.println(F("x-device-key: ********  (held by bridge)"));
  Serial.print(F("x-device-id: ")); Serial.println(DEVICE_ID);
  Serial.print(F("Content-Length: ")); Serial.println(body.length());
  Serial.println(F("Connection: close"));
  Serial.println();
  Serial.println(body);
}

void setup() {
  Serial.begin(9600);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BUZZER, LOW);

  Serial.println(F("================================"));
  Serial.println(F(" SMARTHERD COLLAR (SIMULATED)"));
  Serial.println(F(" TRANSPORT: HTTP/1.1 via bridge"));
  Serial.println(F(" ENDPOINT : /functions/v1/ingest-gps"));
  Serial.println(F(" Keys: j=last payload  p=position  f=fence"));
  Serial.println(F("================================"));
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'j') { Serial.println(F("[last payload]")); Serial.println(lastPayload); }
    else if (c == 'f') {
      Serial.print(F("[fence] centre ")); Serial.print(degStr(FENCE_LAT_E5));
      Serial.print(F(", ")); Serial.print(degStr(FENCE_LON_E5));
      Serial.print(F("  r=")); Serial.print(FENCE_RADIUS_M); Serial.println(F(" m"));
    }
    else if (c == 'p') { Serial.print(F("[seq] ")); Serial.println(seq); }
  }

  if (millis() - lastUpdate < UPLOAD_INTERVAL_MS) return;
  lastUpdate = millis();
  seq++;

  float voltage    = analogRead(TEMP_PIN) * (5.0 / 1023.0);
  float temperatureC = (voltage - 0.5) * 100.0;
  int   pulseRate  = map(analogRead(PULSE_PIN), 0, 1023, 40, 140);
  long  latE5      = map(analogRead(LAT_PIN), 0, 1023, LAT_MIN_E5, LAT_MAX_E5);
  long  lonE5      = map(analogRead(LON_PIN), 0, 1023, LON_MIN_E5, LON_MAX_E5);

  long distanceM   = fenceDistanceM(latE5, lonE5);
  bool insideFence = distanceM <= FENCE_RADIUS_M;

  bool healthAlert = (temperatureC < TEMP_MIN) || (temperatureC > TEMP_MAX) ||
                     (pulseRate < PULSE_MIN)   || (pulseRate > PULSE_MAX);

  lastPayload = buildPayload(latE5, lonE5, temperatureC, pulseRate,
                             insideFence, healthAlert);

  Serial.println();
  Serial.println(F("------ ANIMAL DATA ------"));
  Serial.print(F("Temperature: ")); Serial.print(temperatureC, 1); Serial.println(F(" C"));
  Serial.print(F("Pulse Rate : ")); Serial.print(pulseRate); Serial.println(F(" BPM"));
  Serial.print(F("Latitude   : ")); Serial.println(degStr(latE5));
  Serial.print(F("Longitude  : ")); Serial.println(degStr(lonE5));
  Serial.print(F("Fence dist : ")); Serial.print(distanceM);
  Serial.println(insideFence ? F(" m (INSIDE)") : F(" m (OUTSIDE)"));

  printHttpFrame(lastPayload);

  // The line the bridge parses. One per fix, nothing else on it.
  Serial.print(F("#SH>"));
  Serial.println(lastPayload);

  if (healthAlert || !insideFence) {
    digitalWrite(RED_LED, HIGH);
    digitalWrite(GREEN_LED, LOW);
    tone(BUZZER, 1000, 500);
    Serial.println(healthAlert ? F("ALERT: ABNORMAL PHYSIOLOGY")
                               : F("ALERT: FENCE BREACH"));
    Serial.println(F("SMS: ALERT QUEUED (SIMULATED)"));
  } else {
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
    noTone(BUZZER);
    Serial.println(F("STATUS: MONITORING"));
  }
  Serial.println(F("-------------------------"));
}