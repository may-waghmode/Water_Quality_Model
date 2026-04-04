
## 📁 Project Files

```
water_quality_project/
  ├── water_potability.csv          ← Training dataset (Kaggle)
  ├── water_quality_model.pkl       ← Trained ML model (saved)
  ├── water_quality_80.py           ← Training script (run to retrain)
  ├── water_safety.py               ← Model 1: Safe / Unsafe prediction
  ├── geo_source.py                 ← Model 2: Water source identification
  └── demo.py                       ← Run this to see both models in action
```



## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib
```

### Step 2 — Run the demo
```bash
python demo.py
```

### Step 3 — Use in your own code
```python
from water_safety import predict_safety
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
```
