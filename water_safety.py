"""
FILE 1 — WATER SAFETY PREDICTOR
==================================
What it does: Takes 9 sensor readings → SAFE or NOT SAFE

Needs: water_quality_model.pkl (already trained)

Run: python water_safety.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


def predict_safety(sensor_readings: dict) -> dict:
    """
    Input  — 9 sensor readings as a dict
    Output — safe/unsafe + confidence + reasons

    Example:
        result = predict_safety({
            "ph": 7.2,
            "hardness": 180.0,
            "solids": 15000.0,
            "chloramines": 2.5,
            "sulfate": 200.0,
            "conductivity": 400.0,
            "organic_carbon": 10.0,
            "trihalomethanes": 60.0,
            "turbidity": 3.5
        })
    """

    b    = joblib.load("water_quality_model.pkl")
    orig = b["original_cols"]

    # ── Step 1: Build input with original 9 columns ──────────
    s = pd.DataFrame([{k: sensor_readings.get(k, np.nan) for k in orig}])
    s = pd.DataFrame(b["imputer"].transform(s), columns=orig)

    # ── Step 2: Feature engineering ──────────────────────────
    s["ph_is_safe"]      = ((s["ph"] >= 6.5) & (s["ph"] <= 8.5)).astype(int)
    s["ph_deviation"]    = abs(s["ph"] - 7.0)
    s["chloramine_safe"] = (s["chloramines"] <= 3).astype(int)
    s["turbidity_safe"]  = (s["turbidity"] <= 5).astype(int)
    s["sulfate_safe"]    = (s["sulfate"] <= 250).astype(int)
    s["organic_safe"]    = (s["organic_carbon"] <= 2).astype(int)
    s["thm_safe"]        = (s["trihalomethanes"] <= 80).astype(int)
    flags = ["ph_is_safe", "chloramine_safe", "turbidity_safe",
             "sulfate_safe", "organic_safe", "thm_safe"]
    s["safety_score"]    = s[flags].sum(axis=1)
    s["tds_level"]       = s["solids"].apply(
        lambda v: 0 if v < 300 else 1 if v < 600 else 2 if v < 900 else 3 if v < 1200 else 4
    )
    s["hard_cond_ratio"] = s["hardness"] / (s["conductivity"] + 1e-6)
    s["log_solids"]      = np.log1p(s["solids"])
    s["ph_x_turbidity"]  = s["ph"] * s["turbidity"]
    s["hard_x_sulfate"]  = s["hardness"] * s["sulfate"] / 10000
    d = ["ph_deviation", "turbidity", "organic_carbon", "trihalomethanes", "solids"]
    s["danger_index"]    = (s[d] > s[d].median()).sum(axis=1)

    # ── Step 3: Align, scale, select ─────────────────────────
    for col in b["all_features"]:
        if col not in s.columns:
            s[col] = 0
    s     = s[b["all_features"]]
    s     = b["scaler"].transform(s)
    s     = b["selector"].transform(s)

    # ── Step 4: Predict ───────────────────────────────────────
    proba = b["model"].predict_proba(s)[0]
    pred  = int(proba[1] >= b["threshold"])

    # ── Step 5: Reasons ───────────────────────────────────────
    reasons = []
    r = sensor_readings
    if not (6.5 <= r.get("ph", 7) <= 8.5):
        reasons.append(f"pH {r['ph']:.1f} is outside safe range (6.5 to 8.5)")
    if r.get("turbidity", 0) > 5:
        reasons.append(f"Turbidity {r['turbidity']:.1f} NTU is too high (safe: under 5)")
    if r.get("sulfate", 0) > 250:
        reasons.append(f"Sulfate {r['sulfate']:.0f} mg/L exceeds safe limit (250)")
    if r.get("trihalomethanes", 0) > 80:
        reasons.append(f"Trihalomethanes {r['trihalomethanes']:.0f} exceeds safe limit (80)")
    if r.get("organic_carbon", 0) > 2:
        reasons.append(f"Organic carbon {r['organic_carbon']:.1f} mg/L is elevated")
    if r.get("chloramines", 0) > 3:
        reasons.append(f"Chloramines {r['chloramines']:.1f} mg/L is too high (safe: under 3)")

    return {
        "safe":        bool(pred == 1),
        "label":       "SAFE TO DRINK" if pred == 1 else "NOT SAFE TO DRINK",
        "safe_prob":   f"{proba[1] * 100:.1f}%",
        "unsafe_prob": f"{proba[0] * 100:.1f}%",
        "confidence":  f"{max(proba) * 100:.1f}%",
        "reasons":     reasons if reasons else ["All parameters within safe limits"],
    }


# ══════════════════════════════════════════════════════════
# DEMO — Run to test
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("""
╔══════════════════════════════════════════╗
║   WATER SAFETY PREDICTOR                ║
╚══════════════════════════════════════════╝
""")

    samples = [
        {
            "name": "Clean Tap Water",
            "data": {
                "ph": 7.2, "hardness": 170.0, "solids": 14000.0,
                "chloramines": 2.0, "sulfate": 210.0, "conductivity": 390.0,
                "organic_carbon": 1.5, "trihalomethanes": 55.0, "turbidity": 3.2
            }
        },
        {
            "name": "Contaminated Water",
            "data": {
                "ph": 5.5, "hardness": 280.0, "solids": 40000.0,
                "chloramines": 9.0, "sulfate": 400.0, "conductivity": 680.0,
                "organic_carbon": 22.0, "trihalomethanes": 110.0, "turbidity": 6.8
            }
        },
        {
            "name": "Borderline Water",
            "data": {
                "ph": 6.4, "hardness": 200.0, "solids": 22000.0,
                "chloramines": 4.5, "sulfate": 270.0, "conductivity": 430.0,
                "organic_carbon": 3.5, "trihalomethanes": 75.0, "turbidity": 5.2
            }
        },
    ]

    for sample in samples:
        print(f"  Sample   : {sample['name']}")
        result = predict_safety(sample["data"])
        icon   = "✅" if result["safe"] else "❌"
        print(f"  Result   : {icon} {result['label']}")
        print(f"  Safe     : {result['safe_prob']}  |  Unsafe: {result['unsafe_prob']}")
        print(f"  Confidence: {result['confidence']}")
        if result["reasons"] != ["All parameters within safe limits"]:
            print(f"  Issues:")
            for r in result["reasons"]:
                print(f"    • {r}")
        else:
            print(f"  ✓ {result['reasons'][0]}")
        print()