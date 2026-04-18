import math

def calculate_wqi(readings) -> dict:
    """
    Calculates the Weighted Arithmetic Water Quality Index (WQI).
    
    Standard WHO/BIS Permissible Limits (Si):
    pH: 8.5
    Turbidity: 5 NTU
    DO: 5 mg/L
    BOD: 3 mg/L
    Lead: 0.01 mg/L
    Mercury: 0.001 mg/L
    Arsenic: 0.01 mg/L
    """
    
    # 1. Define Standards (Si) and Ideal Values (Vio)
    standards = {
        'ph': {'Si': 8.5, 'Vio': 7.0},
        'turbidity': {'Si': 5.0, 'Vio': 0.0},
        'do': {'Si': 5.0, 'Vio': 14.6}, # DO standard is min 5, ideal is ~14.6
        'bod': {'Si': 3.0, 'Vio': 0.0},
        'lead': {'Si': 0.01, 'Vio': 0.0},
        'mercury': {'Si': 0.001, 'Vio': 0.0},
        'arsenic': {'Si': 0.01, 'Vio': 0.0}
    }
    
    # 2. Extract values with safe defaults (aligned with standards)
    values = {
        'ph': readings.get("ph", 7.0),
        'turbidity': readings.get("turbidity", readings.get("turbidity_ntu", 1.0)),
        'do': readings.get("do", readings.get("do_mg/l", 7.0)),
        'bod': readings.get("bod", readings.get("bod_mg/l", 1.0)),
        'lead': readings.get("lead", readings.get("lead_mg/l", 0.000)),
        'mercury': readings.get("mercury", readings.get("mercury_mg/l", 0.000)),
        'arsenic': readings.get("arsenic", readings.get("arsenic_mg/l", 0.000))
    }
    
    # 3. Calculate Weight Const (K) -> 1 / sum(1/Si)
    sum_1_over_si = sum(1.0 / s['Si'] for s in standards.values())
    K = 1.0 / sum_1_over_si
    
    # 4. Calculate Sub-Indices (qi) & Unit Weights (Wi)
    wqi_sum = 0
    weights_sum = 0
    parameter_contributions = {}
    
    for param, val in values.items():
        Si = standards[param]['Si']
        Vio = standards[param]['Vio']
        
        # Calculate Unit Weight (Wi)
        Wi = K / Si
        
        # Calculate Sub-index (qi)
        if param == 'do':
            # For DO, higher is better, but formula adapts based on Vio
            # qi = 100 * [ (Vi - Vio) / (Si - Vio) ]
            # standard DO = 5, ideal DO = 14.6
            try:
                qi = 100.0 * (val - Vio) / (Si - Vio)
            except ZeroDivisionError:
                qi = 0
        else:
            try:
                qi = 100.0 * (val - Vio) / (Si - Vio)
            except ZeroDivisionError:
                qi = 0
                
        # Handle negatives / limits
        qi = max(0, qi)
        
        wqi_sum += (qi * Wi)
        weights_sum += Wi
        
        parameter_contributions[param] = qi
        
    # 5. Final WQI Score
    wqi_score = wqi_sum / weights_sum if weights_sum > 0 else 0
    wqi_score = round(wqi_score, 2)
    
    # 6. WQI Categorization
    if wqi_score < 25:
        category = "Excellent (Safe)"
        color = "blue"
    elif wqi_score < 50:
        category = "Good (Acceptable)"
        color = "green"
    elif wqi_score < 75:
        category = "Poor (Needs Treatment)"
        color = "orange"
    elif wqi_score < 100:
        category = "Very Poor (Danger)"
        color = "red"
    else:
        category = "Unsuitable for Drinking (Extreme Pollution)"
        color = "darkred"
        
    return {
        "wqi_score": wqi_score,
        "category": category,
        "color": color,
        "parameter_breakdown": parameter_contributions
    }

if __name__ == "__main__":
    sample = {
        "ph": 7.2, "turbidity": 3.0, "do": 6.5, "bod": 2.0,
        "lead": 0.005, "mercury": 0.0001, "arsenic": 0.002
    }
    result = calculate_wqi(sample)
    print("Normal Sample WQI:", result["wqi_score"], "-", result["category"])
    
    bad_sample = {
        "ph": 4.5, "turbidity": 20.0, "do": 2.5, "bod": 14.0,
        "lead": 0.09, "mercury": 0.006, "arsenic": 0.07
    }
    bad_result = calculate_wqi(bad_sample)
    print("Polluted Sample WQI:", bad_result["wqi_score"], "-", bad_result["category"])
