"""SmartHerd.ai risk engine: turn a model prediction into something a farmer can act on.

A class label is not an alert. "illness_suspect, p=0.83" tells a farmer
nothing they can use. This module does three things the classifier cannot:

  1. Maps the detected behavioural syndrome onto the SADC disease table
     (data/sadc_animal_diseases_2011.csv) to produce a ranked shortlist of
     candidate conditions, together with their zoonotic risk, their control
     measures, and their reported SADC-wide burden.
  2. Escalates rather than replaces. A geofence breach is still a geofence
     breach; the model raises or lowers its priority, it never cancels it.
     That rule exists because a false negative from a model must never be
     able to silence a deterministic alarm.
  3. Routes. A zoonotic notifiable disease goes to the Department of
     Veterinary Services (17755), not just to the farmer's phone.

Nothing here diagnoses an animal. The output is a prompt to look, phrased as
a shortlist of what to rule out, and it says so on every alert.
"""

import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "data", "sadc_animal_diseases_2011.csv")

# Which clinical-sign keywords each detected syndrome is consistent with.
# The model sees fever + hypoactivity + isolation; the CSV knows which
# diseases present that way. This is the join.
SYNDROME_SIGNS = {
    "illness_suspect": {
        "required_any": ["fever", "lameness", "reduced grazing", "weakness",
                         "sudden death", "recumbency", "depression", "difficulty"],
        "boost": ["fever", "lameness"],
    },
    # A flight event is usually a predator or an intruder, not a disease.
    # The only reason to surface a disease at all is the small chance of a
    # neurological condition presenting as sudden aggression, so this
    # syndrome REQUIRES a boost keyword. Without that rule, foot-and-mouth
    # matches on the word "salivation" alone and the farmer gets a
    # foot-and-mouth warning because a dog chased the herd.
    "predator_flight": {
        "required_any": ["aggression", "behaviour change", "neurological",
                         "paralysis"],
        "boost": ["behaviour change", "aggression", "neurological"],
        "require_boost": True,
    },
}

# Species filter - a goat collar should not surface African horse sickness.
SPECIES_TERMS = {
    "goat": ["goat", "small ruminant", "sheep", "caprine", "ovine", "ruminant", "herbivore"],
    "sheep": ["sheep", "small ruminant", "goat", "ovine", "caprine", "ruminant", "herbivore"],
    "cattle": ["cattle", "bovine", "ruminant", "herbivore", "bull", "calf", "cow"],
}

DVS_CALL_CENTRE = "17755"
BAITS_URL = "https://baits3.gov.bw/"


def load_disease_table(path=DEFAULT_CSV):
    df = pd.read_csv(path)
    for c in ["Key_Properties_and_Signs", "Main_Animals_Affected", "Disease",
              "Zoonotic_Risk", "Treatment_or_Control", "Prevention_Methods",
              "Botswana_Wildlife_Context_Species"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def _is_zoonotic(value: str) -> bool:
    v = value.strip().lower()
    return v.startswith("yes") or "zoonotic if" in v or "foodborne" in v


def shortlist_diseases(df, syndrome, species="goat", top_k=5,
                       at_wildlife_interface=False):
    """Rank candidate diseases consistent with a detected syndrome."""
    cfg = SYNDROME_SIGNS.get(syndrome)
    if cfg is None:
        return []

    species_terms = SPECIES_TERMS.get(species.lower(), [species.lower()])
    rows = []
    for _, r in df.iterrows():
        signs = r["Key_Properties_and_Signs"].lower()
        affected = r["Main_Animals_Affected"].lower()

        if not any(t in affected for t in species_terms):
            continue
        hits = [k for k in cfg["required_any"] if k in signs]
        if not hits:
            continue
        boost_hits = [k for k in cfg["boost"] if k in signs]
        if cfg.get("require_boost") and not boost_hits:
            continue

        score = float(len(hits))
        score += 1.5 * len(boost_hits)

        # Burden prior: a disease that actually occurs in the region is a
        # better first guess than one that is merely possible. log-scaled so
        # a single huge outbreak year cannot dominate the ranking.
        outbreaks = pd.to_numeric(r.get("SADC_2011_Outbreaks"), errors="coerce")
        if pd.notna(outbreaks) and outbreaks > 0:
            score += min(2.5, 0.45 * (float(outbreaks) ** 0.35))

        zoon = _is_zoonotic(r["Zoonotic_Risk"])
        if zoon:
            score += 2.0        # human health outranks herd economics

        wild = r["Botswana_Wildlife_Context_Species"].strip()
        if at_wildlife_interface and wild:
            score += 1.5        # buffalo/FMD interface is the Botswana reality

        rows.append({
            "disease": r["Disease"],
            "type": r.get("Disease_Type", ""),
            "score": round(score, 2),
            "signs": r["Key_Properties_and_Signs"],
            "zoonotic": zoon,
            "zoonotic_note": r["Zoonotic_Risk"],
            "control": r["Treatment_or_Control"],
            "prevention": r["Prevention_Methods"],
            "sadc_2011_outbreaks": None if pd.isna(outbreaks) else int(outbreaks),
            "wildlife_hosts_bw": wild,
        })

    rows.sort(key=lambda d: (-d["score"], d["disease"]))
    return rows[:top_k]


def assess(prediction, probability, *, species="goat", animal_id="goat_001",
           fence_breached=False, at_wildlife_interface=False,
           night=False, disease_csv=DEFAULT_CSV, threshold=0.60):
    """Produce a routed, explained alert. Returns a dict, never raises on
    an unknown class."""
    df = load_disease_table(disease_csv)

    severity, channel, action = "info", ["app"], "No action needed."
    shortlist = []

    if fence_breached:
        severity, channel = "high", ["app", "sms"]
        action = "Animal is outside the grazing boundary. Locate and return it."

    if probability < threshold:
        # Below threshold the model stays quiet, but it cannot mute the fence.
        return {
            "animal_id": animal_id, "prediction": prediction,
            "probability": round(float(probability), 3),
            "suppressed": True,
            "severity": severity, "channels": channel, "action": action,
            "disclaimer": "Behaviour model confidence below alert threshold; "
                          "deterministic fence logic is unaffected.",
        }

    if prediction == "illness_suspect":
        severity = "high"
        channel = ["app", "sms"]
        shortlist = shortlist_diseases(df, "illness_suspect", species,
                                       at_wildlife_interface=at_wildlife_interface)
        action = ("Separate this animal from the herd and inspect mouth, feet "
                  "and udder today. Do not move it to another kraal or sell it "
                  "until it has been checked.")
        if any(d["zoonotic"] for d in shortlist):
            severity = "critical"
            channel = ["app", "sms", "dvs"]
            action += (f" Some conditions on this shortlist can infect people. "
                       f"Do not open any carcass. Call DVS on {DVS_CALL_CENTRE}.")

    elif prediction == "theft_suspect":
        severity = "critical"
        channel = ["app", "sms"]
        action = ("Sustained directed movement away from the kraal"
                  + (" at night" if night else "")
                  + ". Confirm the animal's position before approaching, and "
                    "report to the police and your BAITS record if it is moving "
                    f"off the farm. {BAITS_URL}")

    elif prediction == "predator_flight":
        severity = "high"
        channel = ["app", "sms"]
        shortlist = shortlist_diseases(df, "predator_flight", species,
                                       at_wildlife_interface=at_wildlife_interface)
        action = ("Sudden flight by part of the herd. Check for predators, "
                  "dogs or people at the grazing area, and count the herd.")

    elif prediction in ("grazing", "resting"):
        action = "Normal behaviour." if not fence_breached else action

    return {
        "animal_id": animal_id,
        "prediction": prediction,
        "probability": round(float(probability), 3),
        "suppressed": False,
        "severity": severity,
        "channels": channel,
        "action": action,
        "candidate_conditions": shortlist,
        "fence_breached": fence_breached,
        "disclaimer": ("SmartHerd.ai flags behaviour, it does not diagnose. The list "
                       "above is what to rule out, ranked by regional burden "
                       "and human risk. A veterinary officer decides."),
        "sources": ["SADC Animal Disease data 2011 annex",
                    "Statistics Botswana Wildlife Digest 2014 (wildlife context)",
                    "WOAH / MSD Veterinary Manual (clinical summaries)"],
    }


if __name__ == "__main__":
    demo = assess("illness_suspect", 0.91, species="goat",
                  at_wildlife_interface=True, fence_breached=False)
    print(json.dumps(demo, indent=2))
