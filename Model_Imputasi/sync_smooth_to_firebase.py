import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from pathlib import Path

cred_path = Path("d:/Github/Projek_Rainfall/staklimjerukagung-b852a12a367e.json")
csv_path = Path("d:/Github/Projek_Rainfall/Model_Imputasi/outputs_rekonstruksi/id03_smoothed_physics_20260902.csv")

if not firebase_admin._apps:
    cred = credentials.Certificate(str(cred_path))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://staklimjerukagung-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

ref = db.reference('/auto_weather_stat/id-03/data')

print("=== SYNCING SMOOTHED PHYSICS-INFORMED DATASET TO FIREBASE ===")
df = pd.read_csv(csv_path)

# Only update the evening window (16:14 to 20:20 WIB) where the gap and transitions were smoothed
# 16:14 WIB = 1788340440, 20:20 WIB = 1788355200
df_update = df[(df['timestamp'] >= 1788340440) & (df['timestamp'] <= 1788355200)].copy()

payload = {}
for idx, row in df_update.iterrows():
    ts = int(row['timestamp'])
    record = {
        'dew': round(float(row['dew']), 2),
        'humidity': round(float(row['humidity']), 2),
        'pressure': round(float(row['pressure']), 2),
        'rainfall': 0.0,
        'rainrate': 0.0,
        'temperature': round(float(row['temperature']), 2),
        'timestamp': ts,
        'volt': round(float(row['volt']), 2)
    }
    payload[str(ts)] = record

print(f"Updating {len(payload)} smoothed records in Firebase...")
ref.update(payload)
print("[OK] Firebase updated with smoothed physics-informed records!")
