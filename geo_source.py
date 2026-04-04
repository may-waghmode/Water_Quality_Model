"""
FILE 2 — GEO SOURCE PREDICTOR
================================
What it does: Takes chemical readings → predicts WHERE water came from

Source types:
  🏭 Industrial Zone
  🌾 Agricultural Runoff
  🌊 River / Surface Water
  🏔 Mountain Spring / Groundwater
  🏙 Municipal / Treated Supply

No ML model needed — uses WHO and environmental science rules.
No extra files needed.

Run: python geo_source.py
"""


def predict_source(readings: dict) -> dict:
    """
    Input  — chemical readings as a dict (any combination)
    Output — source type + confidence + evidence + recommended action

    Supported keys:
        ph, turbidity, temperature, bod, do,
        lead, mercury, arsenic,
        hardness, solids, conductivity,
        chloramines, sulfate, organic_carbon, trihalomethanes

    Example:
        result = predict_source({
            "ph": 7.2,
            "turbidity": 3.5,
            "lead": 0.002,
            "mercury": 0.0001,
            "arsenic": 0.003,
            "bod": 1.5,
            "temperature": 22.0
        })
    """

    # ── Extract values with safe defaults ────────────────────
    ph           = readings.get("ph",           7.0)
    turbidity    = readings.get("turbidity",    readings.get("turbidity_ntu", 3.0))
    temperature  = readings.get("temperature",  readings.get("temperature_c",
                                readings.get("temperature_°c", 25.0)))
    bod          = readings.get("bod",          readings.get("bod_mg/l", 3.0))
    do_val       = readings.get("do",           readings.get("do_mg/l", 7.0))
    lead         = readings.get("lead",         readings.get("lead_mg/l", 0.005))
    mercury      = readings.get("mercury",      readings.get("mercury_mg/l", 0.0005))
    arsenic      = readings.get("arsenic",      readings.get("arsenic_mg/l", 0.005))
    conductivity = readings.get("conductivity", 400.0)
    chloramines  = readings.get("chloramines",  0.0)
    sulfate      = readings.get("sulfate",      250.0)
    organic_c    = readings.get("organic_carbon", 10.0)

    # ── Score each source type ────────────────────────────────
    scores   = {
        "Industrial Zone":              0,
        "Agricultural Runoff":          0,
        "River / Surface Water":        0,
        "Mountain Spring / Groundwater":0,
        "Municipal / Treated Supply":   0,
    }
    evidence = {k: [] for k in scores}

    # ─────────────────────────────────────────────────────────
    # INDUSTRIAL ZONE
    # Signs: very high heavy metals, acidic pH, high conductivity
    # ─────────────────────────────────────────────────────────
    if lead > 0.05:
        scores["Industrial Zone"] += 35
        evidence["Industrial Zone"].append(
            f"Lead {lead:.3f} mg/L — dangerously high (safe limit: 0.01)"
        )
    elif lead > 0.02:
        scores["Industrial Zone"] += 22
        evidence["Industrial Zone"].append(
            f"Lead {lead:.3f} mg/L — above safe limit (0.01)"
        )
    elif lead > 0.01:
        scores["Industrial Zone"] += 10
        evidence["Industrial Zone"].append(
            f"Lead {lead:.3f} mg/L — slightly above limit"
        )

    if mercury > 0.002:
        scores["Industrial Zone"] += 30
        evidence["Industrial Zone"].append(
            f"Mercury {mercury:.4f} mg/L — highly elevated (safe limit: 0.001)"
        )
    elif mercury > 0.001:
        scores["Industrial Zone"] += 15
        evidence["Industrial Zone"].append(
            f"Mercury {mercury:.4f} mg/L — at the safety limit"
        )

    if arsenic > 0.05:
        scores["Industrial Zone"] += 30
        evidence["Industrial Zone"].append(
            f"Arsenic {arsenic:.3f} mg/L — dangerously high (safe limit: 0.01)"
        )
    elif arsenic > 0.01:
        scores["Industrial Zone"] += 15
        evidence["Industrial Zone"].append(
            f"Arsenic {arsenic:.3f} mg/L — above safe limit (0.01)"
        )

    if ph < 5.0:
        scores["Industrial Zone"] += 28
        evidence["Industrial Zone"].append(
            f"pH {ph:.1f} — highly acidic (industrial waste signature)"
        )
    elif ph < 6.0:
        scores["Industrial Zone"] += 15
        evidence["Industrial Zone"].append(
            f"pH {ph:.1f} — acidic (possible industrial influence)"
        )

    if conductivity > 700:
        scores["Industrial Zone"] += 15
        evidence["Industrial Zone"].append(
            f"Conductivity {conductivity:.0f} µS/cm — very high (dissolved industrial chemicals)"
        )

    # ─────────────────────────────────────────────────────────
    # AGRICULTURAL RUNOFF
    # Signs: high turbidity, high BOD, slightly acidic, pesticide traces
    # ─────────────────────────────────────────────────────────
    if turbidity > 20:
        scores["Agricultural Runoff"] += 28
        evidence["Agricultural Runoff"].append(
            f"Turbidity {turbidity:.1f} NTU — very high (heavy soil/sediment runoff)"
        )
    elif turbidity > 12:
        scores["Agricultural Runoff"] += 18
        evidence["Agricultural Runoff"].append(
            f"Turbidity {turbidity:.1f} NTU — elevated (field runoff)"
        )
    elif turbidity > 8:
        scores["Agricultural Runoff"] += 8
        evidence["Agricultural Runoff"].append(
            f"Turbidity {turbidity:.1f} NTU — moderately elevated"
        )

    if bod > 8:
        scores["Agricultural Runoff"] += 25
        evidence["Agricultural Runoff"].append(
            f"BOD {bod:.1f} mg/L — high organic load (fertilizer/manure runoff)"
        )
    elif bod > 5:
        scores["Agricultural Runoff"] += 12
        evidence["Agricultural Runoff"].append(
            f"BOD {bod:.1f} mg/L — moderate organic load"
        )

    if 5.5 <= ph < 6.5:
        scores["Agricultural Runoff"] += 15
        evidence["Agricultural Runoff"].append(
            f"pH {ph:.1f} — slightly acidic (fertilizer acidification)"
        )

    if sulfate > 350:
        scores["Agricultural Runoff"] += 12
        evidence["Agricultural Runoff"].append(
            f"Sulfate {sulfate:.0f} mg/L — elevated (fertilizer runoff)"
        )

    if arsenic > 0.005 and lead < 0.02:
        scores["Agricultural Runoff"] += 10
        evidence["Agricultural Runoff"].append(
            "Low-level arsenic with moderate lead — pesticide residues possible"
        )

    # ─────────────────────────────────────────────────────────
    # RIVER / SURFACE WATER
    # Signs: medium turbidity, moderate BOD, variable temperature
    # ─────────────────────────────────────────────────────────
    if 5 <= turbidity <= 15:
        scores["River / Surface Water"] += 22
        evidence["River / Surface Water"].append(
            f"Turbidity {turbidity:.1f} NTU — typical river/stream range"
        )

    if 20 <= temperature <= 32:
        scores["River / Surface Water"] += 12
        evidence["River / Surface Water"].append(
            f"Temperature {temperature:.1f}°C — typical open surface water"
        )

    if 3 <= bod <= 8:
        scores["River / Surface Water"] += 18
        evidence["River / Surface Water"].append(
            f"BOD {bod:.1f} mg/L — natural organic matter level"
        )

    if 4 <= do_val <= 7:
        scores["River / Surface Water"] += 12
        evidence["River / Surface Water"].append(
            f"Dissolved oxygen {do_val:.1f} mg/L — normal river range"
        )

    if 6.5 <= ph <= 8.2:
        scores["River / Surface Water"] += 10
        evidence["River / Surface Water"].append(
            f"pH {ph:.1f} — typical river pH"
        )

    # ─────────────────────────────────────────────────────────
    # MOUNTAIN SPRING / GROUNDWATER
    # Signs: very clear water, cold, low BOD, low metals
    # ─────────────────────────────────────────────────────────
    if turbidity < 1:
        scores["Mountain Spring / Groundwater"] += 38
        evidence["Mountain Spring / Groundwater"].append(
            f"Turbidity {turbidity:.1f} NTU — crystal clear (natural filtration through rock/soil)"
        )
    elif turbidity < 3:
        scores["Mountain Spring / Groundwater"] += 22
        evidence["Mountain Spring / Groundwater"].append(
            f"Turbidity {turbidity:.1f} NTU — very low (groundwater quality)"
        )

    if bod < 2:
        scores["Mountain Spring / Groundwater"] += 22
        evidence["Mountain Spring / Groundwater"].append(
            f"BOD {bod:.1f} mg/L — very low (minimal organic matter)"
        )
    elif bod < 3:
        scores["Mountain Spring / Groundwater"] += 12
        evidence["Mountain Spring / Groundwater"].append(
            f"BOD {bod:.1f} mg/L — low organic load"
        )

    if lead < 0.005 and mercury < 0.0005 and arsenic < 0.005:
        scores["Mountain Spring / Groundwater"] += 22
        evidence["Mountain Spring / Groundwater"].append(
            "Heavy metals all very low — no industrial contamination nearby"
        )

    if temperature < 18:
        scores["Mountain Spring / Groundwater"] += 18
        evidence["Mountain Spring / Groundwater"].append(
            f"Temperature {temperature:.1f}°C — cool (underground/mountain source)"
        )
    elif temperature >= 18 and temperature <= 28:
        # Room temperature water is unlikely to be a mountain spring
        scores["Mountain Spring / Groundwater"] = max(
            0, scores["Mountain Spring / Groundwater"] - 15
        )

    if 7.0 <= ph <= 8.5:
        scores["Mountain Spring / Groundwater"] += 10
        evidence["Mountain Spring / Groundwater"].append(
            f"pH {ph:.1f} — neutral to alkaline (mineral-rich rock)"
        )

    # ─────────────────────────────────────────────────────────
    # MUNICIPAL / TREATED SUPPLY
    # Signs: chloramines present, low turbidity, controlled pH
    # Chloramines are the STRONGEST signal — only added in treatment plants
    # ─────────────────────────────────────────────────────────
    if chloramines > 0.5:
        scores["Municipal / Treated Supply"] += 60  # boosted — definitive sign
        evidence["Municipal / Treated Supply"].append(
            f"Chloramines {chloramines:.1f} mg/L — disinfectant added (definitive treatment sign)"
        )
        # Also suppress Mountain Spring when chloramines detected
        scores["Mountain Spring / Groundwater"] = max(
            0, scores["Mountain Spring / Groundwater"] - 30
        )
    elif chloramines > 0.2:
        scores["Municipal / Treated Supply"] += 35
        evidence["Municipal / Treated Supply"].append(
            f"Chloramines {chloramines:.1f} mg/L — trace disinfectant (treated supply)"
        )

    if turbidity < 2 and bod < 3:
        scores["Municipal / Treated Supply"] += 22
        evidence["Municipal / Treated Supply"].append(
            "Low turbidity + low BOD — consistent with treated water"
        )

    if 6.5 <= ph <= 8.5:
        scores["Municipal / Treated Supply"] += 10
        evidence["Municipal / Treated Supply"].append(
            f"pH {ph:.1f} — within controlled municipal range"
        )

    if lead < 0.01 and mercury < 0.001:
        scores["Municipal / Treated Supply"] += 10
        evidence["Municipal / Treated Supply"].append(
            "Heavy metals within limits — treatment effective"
        )

    if 200 <= conductivity <= 600:
        scores["Municipal / Treated Supply"] += 8
        evidence["Municipal / Treated Supply"].append(
            f"Conductivity {conductivity:.0f} µS/cm — typical treated water range"
        )

    # ── Pick winner ───────────────────────────────────────────
    best   = max(scores, key=lambda k: scores[k])
    total  = sum(scores.values())
    conf   = min((scores[best] / total * 100) if total > 0 else 30.0, 95.0)

    if total == 0 or scores[best] < 10:
        best = "River / Surface Water"
        conf = 35.0

    # ── Source info ───────────────────────────────────────────
    info = {
        "Industrial Zone": {
            "icon":   "🏭",
            "action": "DANGEROUS — Do not use. Report to pollution control board.",
        },
        "Agricultural Runoff": {
            "icon":   "🌾",
            "action": "UNSAFE — Contains pesticide residues. Needs treatment.",
        },
        "River / Surface Water": {
            "icon":   "🌊",
            "action": "NEEDS TREATMENT — Filter and disinfect before drinking.",
        },
        "Mountain Spring / Groundwater": {
            "icon":   "🏔",
            "action": "LIKELY SAFE — Low contamination. Basic testing recommended.",
        },
        "Municipal / Treated Supply": {
            "icon":   "🏙",
            "action": "GENERALLY SAFE — Monitor chlorine levels and pipe condition.",
        },
    }

    # ── Ranked list ───────────────────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_pct = [
        (name, f"{(s / total * 100):.0f}%" if total > 0 else "0%")
        for name, s in ranked
    ]

    return {
        "source_type":   best,
        "icon":          info[best]["icon"],
        "confidence":    f"{conf:.0f}%",
        "action":        info[best]["action"],
        "evidence":      evidence[best],
        "all_scores":    ranked_pct,
    }


# ══════════════════════════════════════════════════════════
# DEMO — Run to test
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("""
╔══════════════════════════════════════════╗
║   GEO SOURCE PREDICTOR                  ║
╚══════════════════════════════════════════╝
""")

    samples = [
        {
            "name": "Factory discharge",
            "data": {
                "ph": 4.5, "turbidity": 20.0, "temperature": 36.0,
                "bod": 14.0, "do": 2.5,
                "lead": 0.09, "mercury": 0.006, "arsenic": 0.07,
                "conductivity": 820.0, "chloramines": 0.0,
                "sulfate": 430.0, "organic_carbon": 28.0
            }
        },
        {
            "name": "Farm field drainage",
            "data": {
                "ph": 6.1, "turbidity": 25.0, "temperature": 23.0,
                "bod": 10.0, "do": 4.2,
                "lead": 0.007, "mercury": 0.0007, "arsenic": 0.014,
                "conductivity": 510.0, "chloramines": 0.0,
                "sulfate": 390.0, "organic_carbon": 19.0
            }
        },
        {
            "name": "Mountain spring",
            "data": {
                "ph": 7.7, "turbidity": 0.3, "temperature": 11.0,
                "bod": 0.7, "do": 9.8,
                "lead": 0.001, "mercury": 0.0001, "arsenic": 0.001,
                "conductivity": 260.0, "chloramines": 0.0,
                "sulfate": 130.0, "organic_carbon": 1.1
            }
        },
        {
            "name": "City tap water",
            "data": {
                "ph": 7.4, "turbidity": 0.9, "temperature": 21.0,
                "bod": 1.4, "do": 8.2,
                "lead": 0.003, "mercury": 0.0002, "arsenic": 0.002,
                "conductivity": 370.0, "chloramines": 2.8,
                "sulfate": 195.0, "organic_carbon": 1.7
            }
        },
        {
            "name": "River water",
            "data": {
                "ph": 7.1, "turbidity": 8.5, "temperature": 27.0,
                "bod": 5.0, "do": 5.8,
                "lead": 0.005, "mercury": 0.0005, "arsenic": 0.006,
                "conductivity": 440.0, "chloramines": 0.0,
                "sulfate": 255.0, "organic_carbon": 11.0
            }
        },
    ]

    for sample in samples:
        result = predict_source(sample["data"])
        print(f"  Sample     : {sample['name']}")
        print(f"  Source     : {result['icon']} {result['source_type']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Action     : {result['action']}")
        print(f"  Evidence:")
        for e in result["evidence"]:
            print(f"    • {e}")
        print(f"  All scores:")
        for name, pct in result["all_scores"]:
            bar = "█" * max(1, int(float(pct.replace("%", "")) / 5))
            icon = result["icon"] if name == result["source_type"] else "  "
            print(f"    {icon} {name:<34} {pct:>4}  {bar}")
        print()