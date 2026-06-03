import os
import traceback
from datetime import datetime

import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

# =========================
# KONFIGURASI
# =========================
source_cred = credentials.Certificate(
    "D:/staklimjerukagung-firebase-adminsdk-kcfma-e091165a9b.json"
)

if not firebase_admin._apps:
    firebase_admin.initialize_app(source_cred, {
        "databaseURL": "https://staklimjerukagung-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(BASE_DIR, "raw_data_sensor")
os.makedirs(output_folder, exist_ok=True)

station_ids = ["id-01", "id-02", "id-03", "id-04", "id-05"]

DEFAULT_START_READABLE = "01-01-2022 00:00:00"
END_READABLE = "31-12-2026 23:59:59"


def readable_to_unix(date_str):
    return int(datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S").timestamp())


def to_unix_from_any(value):
    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except Exception:
        pass

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None

    return int(ts.timestamp())


def get_last_saved_timestamp(csv_path):
    if not os.path.isfile(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, index_col=0)

        if df.empty:
            return None

        # prioritas: index CSV = firebase unix time
        idx = pd.to_numeric(df.index, errors="coerce")
        idx = idx[~pd.isna(idx)]
        if len(idx) > 0:
            return int(idx.max())

        # fallback jika file lama punya kolom timestamp
        if "timestamp" in df.columns:
            ts = df["timestamp"].dropna().apply(to_unix_from_any).dropna()
            if len(ts) > 0:
                return int(ts.max())

        return None

    except Exception:
        return None


def fetch_firebase_data(station, start_ts, end_ts):
    ref_path = f"/auto_weather_stat/{station}/data"
    ref_data = db.reference(ref_path)

    results = (
        ref_data.order_by_key()
        .start_at(str(start_ts))
        .end_at(str(end_ts))
        .get()
    )

    if not results:
        return pd.DataFrame()

    # biarkan key Firebase menjadi index
    df = pd.DataFrame.from_dict(results, orient="index")

    # pastikan index unix integer
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.loc[~pd.isna(df.index)].copy()
    df.index = df.index.astype(int)

    # tambahkan kolom sesuai permintaan
    df["timestamp"] = df.index.astype(int)
    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="s", utc=True
    ).dt.tz_convert("Asia/Jakarta")

    return df


def save_station_data(csv_path, df_new):
    if os.path.isfile(csv_path):
        df_old = pd.read_csv(csv_path, index_col=0)
    else:
        df_old = pd.DataFrame()

    df_all = pd.concat([df_old, df_new], axis=0)

    # index Firebase dipakai sebagai acuan utama
    df_all.index = pd.to_numeric(df_all.index, errors="coerce")
    df_all = df_all.loc[~pd.isna(df_all.index)].copy()
    df_all.index = df_all.index.astype(int)

    if "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_numeric(df_all["timestamp"], errors="coerce")
        df_all = df_all.dropna(subset=["timestamp"])
        df_all["timestamp"] = df_all["timestamp"].astype(int)

    df_all = df_all[~df_all.index.duplicated(keep="last")]
    df_all = df_all.sort_index()

    df_all.index.name = "firebase_key"
    df_all.to_csv(csv_path, index=True)
    return df_all


# =========================
# PROSES UTAMA
# =========================
end_ts = readable_to_unix(END_READABLE)

print("Memulai proses pengambilan data dari Firebase...")
print("=" * 70)

for station in station_ids:
    try:
        csv_path = os.path.join(output_folder, f"{station}.csv")

        last_ts = get_last_saved_timestamp(csv_path)
        if last_ts is None:
            start_ts = readable_to_unix(DEFAULT_START_READABLE)
        else:
            start_ts = last_ts + 1

        print(f"\nStasiun: {station}")
        print(f"Start  : {start_ts}")
        print(f"End    : {end_ts}")

        df_new = fetch_firebase_data(station, start_ts, end_ts)

        if df_new.empty:
            print("Tidak ada data baru.")
            continue

        df_final = save_station_data(csv_path, df_new)

        print(f"✅ Tersimpan: {csv_path}")
        print(f"   Total baris: {len(df_final)}")

    except Exception as e:
        print(f"❌ Error pada {station}: {e}")
        print(traceback.format_exc())

print("\nSelesai.")