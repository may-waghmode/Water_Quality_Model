🚀 How to Run
Step 1 — Install dependencies
bashpip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib

Step 2 — Run the demo
bashpython demo.py

Step 3 — Use in your own code
pythonfrom water_safety import predict_safety
from geo_source   import predict_source

# Safety prediction
result = predict_safety({
    "ph": 7.2,
    "hardness": 180.0,
    "solids": 15000.0,
    "chloramines": 2.5,
    "sulfate": 200.0,
    "conductivity": 400.0,
    "organic_carbon": 1.5,
    "trihalomethanes": 55.0,
    "turbidity": 3.2
})
print(result["label"])       # SAFE TO DRINK
print(result["confidence"])  # 84.2%

# Geo source prediction
source = predict_source({
    "ph": 7.2,
    "turbidity": 3.2,
    "temperature": 22.0,
    "bod": 1.5,
    "lead": 0.003,
    "mercury": 0.0002,
    "arsenic": 0.002,
    "chloramines": 2.5
})
print(source["source_type"])  # Municipal / Treated Supply
print(source["confidence"])   # 61%
