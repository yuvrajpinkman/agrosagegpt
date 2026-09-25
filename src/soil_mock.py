"""IoT Simulation Module for 7-in-1 Soil Sensor Data.

Simulates sensor telemetry: Nitrogen (N), Phosphorus (P), Potassium (K),
pH, Moisture, Temperature, and Electrical Conductivity (EC).
"""

from datetime import datetime, timedelta
import random
import pandas as pd


def generate_soil_history(hours: int = 24, freq_mins: int = 60) -> pd.DataFrame:
    """Generate synthetic time-series DataFrame of 7-in-1 soil sensor readings.
    
    Args:
        hours: Number of past hours to generate readings for.
        freq_mins: Interval in minutes between readings.
        
    Returns:
        pandas.DataFrame with timestamped soil parameters.
    """
    total_steps = int((hours * 60) / freq_mins)
    now = datetime.now()
    start_time = now - timedelta(hours=hours)
    
    timestamps = [start_time + timedelta(minutes=i * freq_mins) for i in range(total_steps + 1)]
    
    # Baseline realistic agronomic parameters with small natural variance
    base_n = 45.0   # mg/kg (moderate nitrogen)
    base_p = 22.0   # mg/kg (adequate phosphorus)
    base_k = 180.0  # mg/kg (good potassium)
    base_ph = 6.4   # slightly acidic (ideal for many crops)
    base_moist = 42.0  # % moisture
    base_temp = 26.5   # °C
    base_ec = 1.2      # mS/cm (electrical conductivity)
    
    records = []
    for ts in timestamps:
        # Slight diurnal cycle for temperature & moisture
        hour_factor = (ts.hour - 12) / 12.0  # peaks near midday
        temp = round(base_temp + (3.0 * (1 - abs(hour_factor))) + random.uniform(-0.4, 0.4), 1)
        moisture = round(base_moist - (2.5 * (1 - abs(hour_factor))) + random.uniform(-0.8, 0.8), 1)
        moisture = max(5.0, min(95.0, moisture))
        
        n = round(max(5.0, base_n + random.uniform(-4.0, 4.0)), 1)
        p = round(max(2.0, base_p + random.uniform(-2.5, 2.5)), 1)
        k = round(max(10.0, base_k + random.uniform(-10.0, 10.0)), 1)
        ph = round(max(4.0, min(9.0, base_ph + random.uniform(-0.15, 0.15))), 2)
        ec = round(max(0.1, base_ec + random.uniform(-0.08, 0.08)), 2)
        
        records.append({
            "timestamp": ts,
            "nitrogen_mg_kg": n,
            "phosphorus_mg_kg": p,
            "potassium_mg_kg": k,
            "ph": ph,
            "moisture_pct": moisture,
            "temperature_c": temp,
            "ec_ms_cm": ec
        })
        
    df = pd.DataFrame(records)
    return df


def get_latest_soil_reading(df: pd.DataFrame = None, as_dict: bool = False):
    """Retrieve the latest soil sensor reading.
    
    Args:
        df: Optional existing DataFrame. If None, generates fresh synthetic data.
        as_dict: If True, returns a dictionary. If False, returns formatted string.
        
    Returns:
        dict or str: Latest sensor reading and agronomic status summary.
    """
    if df is None or df.empty:
        df = generate_soil_history(hours=24)
        
    latest = df.iloc[-1].to_dict()
    
    # Assess status for farmers
    status = {
        "timestamp": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(latest["timestamp"], datetime) else str(latest["timestamp"]),
        "nitrogen": {"value": latest["nitrogen_mg_kg"], "unit": "mg/kg", "status": "Medium" if 30 <= latest["nitrogen_mg_kg"] <= 60 else ("Low" if latest["nitrogen_mg_kg"] < 30 else "High")},
        "phosphorus": {"value": latest["phosphorus_mg_kg"], "unit": "mg/kg", "status": "Optimal" if 20 <= latest["phosphorus_mg_kg"] <= 50 else ("Low" if latest["phosphorus_mg_kg"] < 20 else "High")},
        "potassium": {"value": latest["potassium_mg_kg"], "unit": "mg/kg", "status": "Optimal" if 150 <= latest["potassium_mg_kg"] <= 250 else ("Low" if latest["potassium_mg_kg"] < 150 else "High")},
        "ph": {"value": latest["ph"], "unit": "pH", "status": "Optimal (6.0 - 7.5)" if 6.0 <= latest["ph"] <= 7.5 else ("Acidic" if latest["ph"] < 6.0 else "Alkaline")},
        "moisture": {"value": latest["moisture_pct"], "unit": "%", "status": "Adequate" if 35 <= latest["moisture_pct"] <= 65 else ("Dry" if latest["moisture_pct"] < 35 else "Saturated")},
        "temperature": {"value": latest["temperature_c"], "unit": "°C", "status": "Optimal" if 20 <= latest["temperature_c"] <= 32 else "Extreme"},
        "ec": {"value": latest["ec_ms_cm"], "unit": "mS/cm", "status": "Normal Salinity" if latest["ec_ms_cm"] <= 2.0 else "High Salinity"}
    }
    
    if as_dict:
        return status
        
    formatted = (
        f"Soil Status (Latest as of {status['timestamp']}):\n"
        f"- Nitrogen (N): {status['nitrogen']['value']} {status['nitrogen']['unit']} ({status['nitrogen']['status']})\n"
        f"- Phosphorus (P): {status['phosphorus']['value']} {status['phosphorus']['unit']} ({status['phosphorus']['status']})\n"
        f"- Potassium (K): {status['potassium']['value']} {status['potassium']['unit']} ({status['potassium']['status']})\n"
        f"- pH: {status['ph']['value']} ({status['ph']['status']})\n"
        f"- Moisture: {status['moisture']['value']}% ({status['moisture']['status']})\n"
        f"- Soil Temperature: {status['temperature']['value']}°C ({status['temperature']['status']})\n"
        f"- Electrical Conductivity (EC): {status['ec']['value']} mS/cm ({status['ec']['status']})"
    )
    return formatted
