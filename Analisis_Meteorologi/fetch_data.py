# 1. SETUP & DEPENDENCIES
import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
import firebase_admin
from firebase_admin import credentials, db
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings
sns.set_theme(style="whitegrid")

# Constants
STATIONS = ['id-02','id-03','id-05']

# DETECT ENVIRONMENT (LOCAL VS KAGGLE)
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')

if IS_KAGGLE:
    import glob
    # Langsung menargetkan folder utama tanpa masuk ke subfolder 'cache_data' lagi
    READ_CACHE_DIR = '/kaggle/input/notebooks/jerismeteo/cek-data-sensor'
    WRITE_CACHE_DIR = '/kaggle/working'
    
    # Fallback fleksibel jika nama folder input di Kaggle sedikit bergeser
    if not os.path.exists(READ_CACHE_DIR):
        candidates = glob.glob('/kaggle/input/**/cek-data-sensor', recursive=True)
        if candidates:
            READ_CACHE_DIR = candidates[0]
            
    display(Markdown(f"> ℹ️ **Kaggle Environment Detected**\n> Read cache: `{READ_CACHE_DIR}`\n> Write cache: `{WRITE_CACHE_DIR}`"))
else:
    # Jalur absolut lokal Windows sesuai direktori komputer Anda
    READ_CACHE_DIR = r"D:\Github\Projek_Rainfall\Analisis_Meteorologi\cache_data"
    WRITE_CACHE_DIR = r"D:\Github\Projek_Rainfall\Analisis_Meteorologi\cache_data"
    display(Markdown(f"> ℹ️ **Local Environment Detected**\n> Read cache: `{READ_CACHE_DIR}`\n> Write cache: `{WRITE_CACHE_DIR}`"))

# Memastikan folder output/write sedia digunakan
os.makedirs(WRITE_CACHE_DIR, exist_ok=True)

# FIREBASE AUTHENTICATION
if not firebase_admin._apps:
    # Dynamic credential resolution
    cert_path = None
    candidates = [
        'D:/staklimjerukagung-firebase-adminsdk-kcfma-e091165a9b.json',
        'staklimjerukagung-firebase-adminsdk-kcfma-e091165a9b.json',
    ]
    # Search for any firebase-adminsdk file in workspace
    import glob
    for p in glob.glob('**/*firebase-adminsdk*.json', recursive=True):
        candidates.append(p)
    for p in glob.glob('../*firebase-adminsdk*.json'):
        candidates.append(p)
    # Search in Kaggle inputs
    if os.path.exists('/kaggle/input'):
        for p in glob.glob('/kaggle/input/**/*firebase-adminsdk*.json', recursive=True):
            candidates.append(p)
            
    # Find the first candidate that exists
    for c in candidates:
        if os.path.exists(c):
            cert_path = c
            break
            
    if cert_path:
        try:
            cred = credentials.Certificate(cert_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://staklimjerukagung-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
            display(Markdown(f"> ✅ **Firebase Initialized Successfully** using `{cert_path}`"))
        except Exception as e:
            display(Markdown(f"> ❌ **Error Initializing Firebase: {e}**"))
    else:
        display(Markdown("> ⚠️ **Warning: Credential file not found! Firebase database connection will not be available.**"))
else:
    display(Markdown("> ✅ **Firebase Already Initialized**"))

# 2. CHUNKED BIG DATA RETRIEVAL ENGINE & INCREMENTAL UPDATE
def fetch_and_cache_data_chunked(station_id, chunk_size=500000):
    write_cache_path = os.path.join(WRITE_CACHE_DIR, f"{station_id}_raw.csv")
    read_cache_path = os.path.join(READ_CACHE_DIR, f"{station_id}_raw.csv")
    
    df_existing = pd.DataFrame()
    last_key = None
    
    # Check where to read from (prefer write cache if updated, otherwise fallback to read cache)
    cache_path_to_read = None
    if os.path.exists(write_cache_path):
        cache_path_to_read = write_cache_path
    elif os.path.exists(read_cache_path):
        cache_path_to_read = read_cache_path
        
    if cache_path_to_read:
        try:
            df_existing = pd.read_csv(cache_path_to_read, low_memory=False)
            if not df_existing.empty:
                max_ts = df_existing['timestamp'].max()
                last_key = str(int(max_ts))
                display(Markdown(f"✅ Loaded `{station_id}` from cache. Fetching new data since `{last_key}`..."))
        except Exception as e:
            display(Markdown(f"⚠️ Failed to read cache `{station_id}`: {e}. Re-downloading full dataset..."))
            df_existing = pd.DataFrame()
            last_key = None
    else:
        display(Markdown(f"⏳ Downloading `{station_id}` from scratch..."))
        
    # If Firebase is not initialized, we cannot download new data. Gracefully return cached data if available.
    if not firebase_admin._apps:
        display(Markdown(f"⚠️ Firebase not initialized. Skipping remote fetch. Returning existing local cache for `{station_id}`."))
        return df_existing
        
    ref = db.reference(f'/auto_weather_stat/{station_id}/data')
    all_data = {}
    total_fetched = 0
    
    while True:
        query = ref.order_by_key().limit_to_first(chunk_size)
        if last_key:
            query = query.start_at(last_key)
            
        try:
            chunk = query.get()
        except Exception as e:
            display(Markdown(f"⚠️ Error fetching `{station_id}`: {e}"))
            break
            
        if not chunk:
            break
            
        keys = list(chunk.keys())
        # Prevent infinite loop by removing the overlapping last_key
        if last_key and keys[0] == last_key:
            del chunk[keys[0]]
            keys = keys[1:]
            
        if not chunk:
            break
            
        all_data.update(chunk)
        last_key = keys[-1]
        total_fetched += len(keys)
        
        # If we received less than chunk_size, we've reached the end
        if len(keys) < (chunk_size - 1):
            break
            
    if not all_data:
        if not df_existing.empty:
            display(Markdown(f"👍 `{station_id}` is up to date! Total records: {len(df_existing):,}"))
            # If it's up to date and not in write directory, copy it to write directory so it can be used there
            if cache_path_to_read == read_cache_path:
                df_existing.to_csv(write_cache_path, index=False)
            return df_existing
        else:
            display(Markdown(f"⚠️ No data found for `{station_id}`."))
            return pd.DataFrame()
            
    # Format new data
    df_new = pd.DataFrame.from_dict(all_data, orient='index')
    
    if 'timestamp' not in df_new.columns:
        df_new['timestamp'] = df_new.index
    else:
        df_new['timestamp'] = df_new['timestamp'].fillna(pd.Series(df_new.index, index=df_new.index))
        
    df_new['timestamp'] = pd.to_numeric(df_new['timestamp'], errors='coerce')
    df_new = df_new.dropna(subset=['timestamp'])
    df_new['timestamp'] = df_new['timestamp'].astype(int)
    
    # Merge existing and new
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    else:
        df_combined = df_new.reset_index(drop=True)
        
    # Perbaiki tipe data campuran menjadi float agar kompatibel dengan Parquet
    for col in df_combined.columns:
        if col != 'timestamp' and df_combined[col].dtype == 'object':
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
    df_combined.to_csv(write_cache_path, index=False)
    display(Markdown(f"💾 Saved `{station_id}` ({total_fetched:,} new records) to cache. Total records: {len(df_combined):,}"))
    return df_combined

dfs = {}
for st in STATIONS:
    df = fetch_and_cache_data_chunked(st)
    if not df.empty:
        dfs[st] = df
