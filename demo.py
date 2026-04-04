"""
WATER QUALITY MONITORING SYSTEM — DEMO
========================================
Run this to see what the system does.

Needs:
  water_quality_model.pkl
  water_safety.py
  geo_source.py

Run: python demo.py
"""

import sys
sys.path.insert(0, ".")

from water_safety import predict_safety
from geo_source  import predict_source


def analyze_water(name, readings):
    """Run both models and print a clean combined report."""

    safety = predict_safety(readings)
    source = predict_source(readings)

    safe_icon = "✅" if safety["safe"] else "❌"

    print("=" * 54)
    print(f"  SAMPLE: {name}")
    print("=" * 54)

    print(f"""
  ┌─────────────────────────────────────────┐
  │  SAFETY CHECK                           │
  │                                         │
  │  Result     : {safe_icon} {safety["label"]:<25}│
  │  Safe prob  : {safety["safe_prob"]:<10}                   │
  │  Confidence : {safety["confidence"]:<10}                   │
  └─────────────────────────────────────────┘""")

    if safety["reasons"] != ["All parameters within safe limits"]:
        print(f"\n  Issues detected:")
        for r in safety["reasons"]:
            print(f"    ⚠  {r}")
    else:
        print(f"\n  ✓  {safety['reasons'][0]}")

    print(f"""
  ┌─────────────────────────────────────────┐
  │  GEO SOURCE                             │
  │                                         │
  │  Source     : {source["icon"]} {source["source_type"]:<27}│
  │  Confidence : {source["confidence"]:<10}                   │
  └─────────────────────────────────────────┘""")

    print(f"\n  Why we think this:")
    for e in source["evidence"][:3]:
        print(f"    •  {e}")

    print(f"\n  Recommended action:")
    print(f"    ➤  {source['action']}")

    print(f"\n  Source probabilities:")
    for sname, pct in source["all_scores"]:
        bar   = "█" * max(1, int(float(pct.replace("%","")) / 5))
        icon  = source["icon"] if sname == source["source_type"] else "  "
        print(f"    {icon} {sname:<34} {pct:>4}  {bar}")

    print()


# ══════════════════════════════════════════════════════════
# TEST SAMPLES — show different water scenarios
# ══════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════╗
║      WATER QUALITY MONITORING SYSTEM                ║
║      ML Model Demo — Both Models Combined           ║
╚══════════════════════════════════════════════════════╝
""")

samples = [
    {
        "name": "Sample A — Clean Municipal Water",
        "readings": {
            "ph": 7.4, "hardness": 160.0, "solids": 12000.0,
            "chloramines": 2.8, "sulfate": 195.0, "conductivity": 370.0,
            "organic_carbon": 1.7, "trihalomethanes": 45.0, "turbidity": 0.9,
            # extra for geo
            "temperature": 21.0, "bod": 1.4, "do": 8.2,
            "lead": 0.003, "mercury": 0.0002, "arsenic": 0.002,
        }
    },
    {
        "name": "Sample B — Industrial Discharge",
        "readings": {
            "ph": 4.5, "hardness": 290.0, "solids": 46000.0,
            "chloramines": 0.0, "sulfate": 430.0, "conductivity": 820.0,
            "organic_carbon": 28.0, "trihalomethanes": 115.0, "turbidity": 20.0,
            # extra for geo
            "temperature": 36.0, "bod": 14.0, "do": 2.5,
            "lead": 0.09, "mercury": 0.006, "arsenic": 0.07,
        }
    },
    {
        "name": "Sample C — Mountain Spring",
        "readings": {
            "ph": 7.7, "hardness": 120.0, "solids": 8000.0,
            "chloramines": 0.0, "sulfate": 130.0, "conductivity": 260.0,
            "organic_carbon": 1.1, "trihalomethanes": 20.0, "turbidity": 0.3,
            # extra for geo
            "temperature": 11.0, "bod": 0.7, "do": 9.8,
            "lead": 0.001, "mercury": 0.0001, "arsenic": 0.001,
        }
    },
    {
        "name": "Sample D — Agricultural Runoff",
        "readings": {
            "ph": 6.1, "hardness": 200.0, "solids": 28000.0,
            "chloramines": 0.0, "sulfate": 390.0, "conductivity": 510.0,
            "organic_carbon": 19.0, "trihalomethanes": 55.0, "turbidity": 25.0,
            # extra for geo
            "temperature": 23.0, "bod": 10.0, "do": 4.2,
            "lead": 0.007, "mercury": 0.0007, "arsenic": 0.014,
        }
    },
    {
        "name": "Sample E — River Water",
        "readings": {
            "ph": 7.1, "hardness": 190.0, "solids": 20000.0,
            "chloramines": 0.0, "sulfate": 255.0, "conductivity": 440.0,
            "organic_carbon": 11.0, "trihalomethanes": 65.0, "turbidity": 8.5,
            # extra for geo
            "temperature": 27.0, "bod": 5.0, "do": 5.8,
            "lead": 0.005, "mercury": 0.0005, "arsenic": 0.006,
        }
    },
]

for sample in samples:
    analyze_water(sample["name"], sample["readings"])

print("=" * 54)
print("  END OF DEMO")
print("=" * 54)
print("""
  This system uses 2 models:
  1. Safety Model  — ML trained on 3276 real water samples
                     88-90% accuracy on test data
  2. Source Model  — Rule-based using WHO guidelines
                     Identifies 5 water source types

  For hardware integration:
    from water_safety import predict_safety
    from geo_source   import predict_source

    result = predict_safety(your_sensor_readings)
    source = predict_source(your_sensor_readings)
""")