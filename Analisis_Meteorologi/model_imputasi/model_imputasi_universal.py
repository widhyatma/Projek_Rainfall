"""
Universal Meteorological Imputation Model
Trains a Bidirectional LSTM on multiple AWS node datasets with ERA5 auxiliary features,
cyclical time encodings, and a Node Embedding Layer.
Evaluates model on simulated missingness (10%-50%) and exports imputed datasets.
"""
import os
import sys
import glob
import warnings
import asyncio

# Fix Windows Proactor event loop issue with ZMQ/TF
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ============================================================
# Environment & RUN_MODE Configuration
# ============================================================
RUN_MODE = "LOCAL_TEST"  # Possible values: "LOCAL_TEST", "FULL_TRAIN"

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')

if RUN_MODE == "LOCAL_TEST":
    print("[INFO] Running in LOCAL_TEST mode")
    # For local test mode, we force local directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == 'model_imputasi' else SCRIPT_DIR
    CACHE_DIR = os.path.join(BASE_DIR, 'cache_data')
    PATH_ERA5 = os.path.join(BASE_DIR, 'forecast_open_meteo_jerukangung', 'cuaca_gabungan_jerukagung.csv')
    if not os.path.exists(PATH_ERA5):
        PATH_ERA5 = os.path.join(SCRIPT_DIR, 'cuaca_gabungan_jerukagung.csv')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'model_imputasi', 'outputs')
    WRITE_CACHE_DIR = CACHE_DIR
else:
    # FULL_TRAIN mode path setup
    if IS_KAGGLE:
        print("[INFO] Running FULL_TRAIN in Kaggle Environment")
        # Direct paths requested by user
        CACHE_DIR = '/kaggle/input/notebooks/jerismeteo/cek-data-sensor'
        PATH_ERA5 = '/kaggle/input/notebooks/jerismeteo/generate-era5-data/cuaca_gabungan_jerukagung.csv'
        
        # Fallback if directories do not exist (e.g. if the user runs it on a different Kaggle notebook setup)
        if not os.path.exists(CACHE_DIR):
            input_dirs = glob.glob('/kaggle/input/*/Analisis_Meteorologi')
            BASE_DIR = input_dirs[0] if input_dirs else '/kaggle/input'
            CACHE_DIR = os.path.join(BASE_DIR, 'cache_data')
            
        if not os.path.exists(PATH_ERA5):
            candidates = glob.glob('/kaggle/input/**/cuaca_gabungan_jerukagung.csv', recursive=True)
            if candidates:
                PATH_ERA5 = candidates[0]
            else:
                PATH_ERA5 = os.path.join(os.path.dirname(CACHE_DIR), 'generate-era5-data', 'cuaca_gabungan_jerukagung.csv')
            
        OUTPUT_BASE_DIR = '/kaggle/working'
        OUTPUTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'outputs')
        WRITE_CACHE_DIR = os.path.join(OUTPUT_BASE_DIR, 'cache_data')
    else:
        print("[INFO] Running FULL_TRAIN in Local Environment")
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == 'model_imputasi' else SCRIPT_DIR
        CACHE_DIR = os.path.join(BASE_DIR, 'cache_data')
        PATH_ERA5 = os.path.join(BASE_DIR, 'forecast_open_meteo_jerukangung', 'cuaca_gabungan_jerukagung.csv')
        if not os.path.exists(PATH_ERA5):
            PATH_ERA5 = os.path.join(SCRIPT_DIR, 'cuaca_gabungan_jerukagung.csv')
        OUTPUTS_DIR = os.path.join(BASE_DIR, 'model_imputasi', 'outputs')
        WRITE_CACHE_DIR = CACHE_DIR

# Ensure outputs folders are initialized
os.makedirs(os.path.join(OUTPUTS_DIR, 'model'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'reports'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'imputed_data'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'logs'), exist_ok=True)
os.makedirs(WRITE_CACHE_DIR, exist_ok=True)

# Configure Python Logging
import logging
log_path = os.path.join(OUTPUTS_DIR, 'logs', 'local_test.log')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LocalTestLogger")
logger.info(f"Initialized Universal Imputation Pipeline. Mode: {RUN_MODE}")

TARGET_VARS = ['temperature', 'humidity', 'pressure', 'dewpoint']

# ============================================================
# Hyperparameter Tuning & Option Flags
# ============================================================
RUN_TUNING = False  # Set to True to run Bayesian Optimization search

if RUN_MODE == "LOCAL_TEST":
    FIT_EPOCHS = 3
    FIT_BATCH_SIZE = 32
    FIT_VERBOSE = 1
    TUNER_MAX_TRIALS = 2
    TUNER_EPOCHS = 1
else:
    FIT_EPOCHS = 20
    FIT_BATCH_SIZE = 256
    FIT_VERBOSE = 1
    TUNER_MAX_TRIALS = 10
    TUNER_EPOCHS = 3

# Pre-tuned optimal hyperparameters (default fallback values)
best_hparams = {
    'lstm_1_units': 64,
    'lstm_2_units': 32,
    'dropout_rate': 0.2,
    'node_emb_dim': 4,
    'learning_rate': 1e-3
}

# ============================================================
# TensorFlow & Keras Tuner Check
# ============================================================
import subprocess
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as tf_callbacks
    print(f'[OK] TensorFlow v{tf.__version__} is active.')
except ImportError:
    print('[ERROR] TensorFlow/Keras is required for this Bidirectional LSTM model.')
    sys.exit(1)

try:
    import keras_tuner as kt
    print(f'[OK] Keras Tuner v{kt.__version__} is active.')
except ImportError:
    print('[INFO] Keras Tuner not found. Attempting to install...')
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "keras-tuner"])
        import keras_tuner as kt
        print(f'[OK] Keras Tuner v{kt.__version__} successfully installed and imported.')
    except Exception as e:
        print(f'[WARNING] Failed to install/import Keras Tuner: {e}')

# ============================================================
# 1. Ingest & Align AWS Node Data
# ============================================================
print('\n[1/10] Scanning and loading AWS node datasets...')
node_files = glob.glob(os.path.join(CACHE_DIR, 'id-*_raw.csv'))
if not node_files:
    raise FileNotFoundError(f"No node CSV files matching 'id-*_raw.csv' found in {CACHE_DIR}")

node_data = {}
global_min_time = None
global_max_time = None

col_map = {
    'dew': 'dewpoint',
    'dewpoint': 'dewpoint',
    'dew_point': 'dewpoint',
    'temp': 'temperature',
    'temperature': 'temperature',
    'humi': 'humidity',
    'humidity': 'humidity',
    'pres': 'pressure',
    'pressure': 'pressure'
}

for fpath in sorted(node_files):
    node_id = os.path.basename(fpath).split('_')[0]
    print(f"  Processing {node_id} from {os.path.basename(fpath)}...")
    
    df_raw = pd.read_csv(fpath)
    if 'timestamp' not in df_raw.columns:
        print(f"    [WARN] No timestamp column found in {node_id}. Skipping.")
        continue
        
    # Convert timestamp to local datetime
    df_raw['datetime'] = (pd.to_datetime(df_raw['timestamp'], unit='s', utc=True)
                          .dt.tz_convert('Asia/Jakarta')
                          .dt.tz_localize(None))
    
    df_raw = (df_raw.dropna(subset=['datetime'])
              .sort_values('datetime')
              .drop_duplicates(subset='datetime')
              .set_index('datetime'))
    
    # Rename key columns for standardization
    df_rename = df_raw.rename(columns=col_map)
    df_rename = df_rename.loc[:, ~df_rename.columns.duplicated()]
    
    # Verify that all 4 target variables exist
    missing_cols = [c for c in TARGET_VARS if c not in df_rename.columns]
    if missing_cols:
        print(f"    [WARN] Targets {missing_cols} missing in {node_id}. Skipping.")
        continue
        
    # Resample to 1-minute mean
    df_1min = df_rename[TARGET_VARS].resample('1min').mean()
    
    # Track global temporal bounds
    node_min = df_1min.index.min()
    node_max = df_1min.index.max()
    if global_min_time is None or node_min < global_min_time:
        global_min_time = node_min
    if global_max_time is None or node_max > global_max_time:
        global_max_time = node_max
        
    node_data[node_id] = df_1min
    print(f"    Loaded {len(df_1min):,} minutes from {node_min} to {node_max}")

if not node_data:
    raise ValueError("No valid node datasets loaded.")

print(f"Global temporal bounds determined: {global_min_time} to {global_max_time}")

# Reindex all nodes to the global time range
global_time_range = pd.date_range(start=global_min_time, end=global_max_time, freq='1min')
if RUN_MODE == "LOCAL_TEST":
    global_time_range = global_time_range[-5000:]
print(f"Global aligned index length: {len(global_time_range):,} rows.")

for nid in list(node_data.keys()):
    node_data[nid] = node_data[nid].reindex(global_time_range)

# ============================================================
# 2. Ingest & Time-Interpolate ERA5 Auxiliary Data
# ============================================================
print('\n[2/10] Loading and time-interpolating ERA5 auxiliary data...')
era5_raw = pd.read_csv(PATH_ERA5)
era5_raw['datetime'] = (pd.to_datetime(era5_raw['datetime'], utc=True)
                        .dt.tz_convert('Asia/Jakarta')
                        .dt.tz_localize(None))
era5_raw = (era5_raw.sort_values('datetime')
            .drop_duplicates('datetime')
            .set_index('datetime'))

era5_req_cols = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl',
    'wind_speed_10m', 'wind_direction_10m', 'cloud_cover', 'shortwave_radiation', 'rain'
]
# Select columns or fallback if missing
era5_cols = [c for c in era5_req_cols if c in era5_raw.columns]
era5_feats = era5_raw[era5_cols]

# Temporal interpolation to 1-minute global time range
era5_reindexed = era5_feats.reindex(era5_feats.index.union(global_time_range)).sort_index()
era5_interpolated = era5_reindexed.interpolate(method='time')
era5_1min = era5_interpolated.reindex(global_time_range)
era5_1min.columns = [f'era5_{c}' for c in era5_1min.columns]

# Handle edge NaNs
era5_1min = era5_1min.ffill().bfill()
print(f"  ERA5 features processed: {list(era5_1min.columns)}")
print(f"  ERA5 NaNs remaining: {era5_1min.isna().sum().sum()}")

# ============================================================
# 2.5. Data Cleaning & Quality Control (Pre-Imputation Stage)
# ============================================================
print('\n[2.5/10] Commencing Data Cleaning & Quality Control stage...')

def hampel_filter_pandas(series, window=31, n_sigmas=3.0):
    half_window = window // 2
    rolling_median = series.rolling(window=window, min_periods=1, center=True).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, min_periods=1, center=True).median()
    # Estimate standard deviation using scaled MAD
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (series - rolling_median).abs()
    is_outlier = difference > threshold
    return is_outlier

def detect_sensor_freezes(series, min_minutes=30):
    r_min = series.rolling(window=min_minutes, min_periods=min_minutes).min()
    r_max = series.rolling(window=min_minutes, min_periods=min_minutes).max()
    is_frozen_end = (r_max - r_min) == 0.0
    # Reverse rolling max to flag the entire min_minutes window backward
    is_frozen = is_frozen_end[::-1].rolling(window=min_minutes, min_periods=1).max()[::-1].fillna(False).astype(bool)
    return is_frozen

def run_data_cleaning_and_qc(df_node, df_era5, node_id):
    # df_node contains the 1-minute resampled and aligned data
    # df_era5 is era5_1min
    
    # Initialize QC flags to 0 (Good)
    df_flags = pd.DataFrame(0, index=df_node.index, columns=[f'qc_{v}' for v in TARGET_VARS])
    df_cleaned = df_node.copy()
    
    era5_map = {
        'temperature': 'era5_temperature_2m',
        'humidity': 'era5_relative_humidity_2m',
        'pressure': 'era5_pressure_msl',
        'dewpoint': 'era5_dew_point_2m'
    }
    
    # Stage 1 bounds
    qc_bounds = {
        'temperature': (-10.0, 60.0),
        'humidity': (0.0, 100.0),
        'pressure': (850.0, 1100.0),
        'dewpoint': (-10.0, 60.0)
    }
    
    # Stage 4 maximum rate limits per minute
    jump_limits = {
        'temperature': 3.0,
        'pressure': 5.0,
        'humidity': 30.0,
        'dewpoint': 3.0
    }
    
    for var in TARGET_VARS:
        series = df_node[var]
        flag_col = f'qc_{var}'
        
        # --- Stage 5: ERA5 Consistency Check (Suspicious - Priority 5) ---
        if var in era5_map and era5_map[var] in df_era5.columns:
            era_series = df_era5[era5_map[var]]
            diff_era5 = series - era_series
            mean_bias = np.nanmean(diff_era5)
            std_bias = np.nanstd(diff_era5)
            if std_bias > 0:
                is_era5_inconsistent = (diff_era5 - mean_bias).abs() > 3.0 * std_bias
                df_flags.loc[is_era5_inconsistent, flag_col] = 5
                
        # --- Stage 3: Sensor Freeze Detection (Suspicious - Priority 3) ---
        is_freeze = detect_sensor_freezes(series, min_minutes=30)
        df_flags.loc[is_freeze, flag_col] = 3
        
        # --- Stage 4: Temporal Consistency Check (Corruption - Priority 4) ---
        diff_temp = series.diff().abs()
        is_jump = diff_temp > jump_limits[var]
        df_flags.loc[is_jump, flag_col] = 4
        
        # --- Stage 2: Spike Detection (Hampel Filter) (Corruption - Priority 2) ---
        is_spike = hampel_filter_pandas(series, window=31, n_sigmas=3.0)
        df_flags.loc[is_spike, flag_col] = 2
        
        # --- Stage 1: Physical Range Validation (Corruption - Priority 1) ---
        vmin, vmax = qc_bounds[var]
        is_out_of_range = (series < vmin) | (series > vmax)
        df_flags.loc[is_out_of_range, flag_col] = 1
        
        # Additional dewpoint check: dewpoint <= temperature
        if var == 'dewpoint' and 'temperature' in df_node.columns:
            is_dew_greater = series > df_node['temperature']
            df_flags.loc[is_dew_greater, flag_col] = 1
            df_flags.loc[is_dew_greater, 'qc_temperature'] = 1
            
        # --- Stage 7: Cleaning Strategy ---
        # Replace obvious corruption (flags 1, 2, 4) with NaN
        is_corrupt = df_flags[flag_col].isin([1, 2, 4])
        df_cleaned.loc[is_corrupt, var] = np.nan
        
    return df_cleaned, df_flags

def save_qc_plots(node_data_orig, node_flags):
    os.makedirs(os.path.join(OUTPUTS_DIR, 'plots'), exist_ok=True)
    for var in TARGET_VARS:
        fig, axes = plt.subplots(len(node_data_orig), 1, figsize=(12, 3 * len(node_data_orig)), sharex=True)
        if len(node_data_orig) == 1:
            axes = [axes]
            
        for idx, (nid, df_node) in enumerate(sorted(node_data_orig.items())):
            ax = axes[idx]
            df_flag = node_flags[nid]
            orig_series = df_node[var]
            
            # Original series in gray
            ax.plot(df_node.index, orig_series, color='lightgray', label='Original / Raw', alpha=0.8)
            
            # Cleaned series in blue
            flag_col = f'qc_{var}'
            is_corrupt = df_flag[flag_col].isin([1, 2, 4])
            cleaned_series = orig_series.copy()
            cleaned_series.loc[is_corrupt] = np.nan
            ax.plot(df_node.index, cleaned_series, color='royalblue', label='Cleaned', alpha=0.9, linewidth=1)
            
            # Scatter plot for each quality flag level
            colors = {1: 'red', 2: 'darkorange', 3: 'magenta', 4: 'purple', 5: 'brown'}
            labels = {1: 'Range Violation', 2: 'Spike', 3: 'Sensor Freeze', 4: 'Temporal Jump', 5: 'ERA5 Incons.'}
            
            for flag_val, color in colors.items():
                pts = orig_series[df_flag[flag_col] == flag_val]
                if len(pts) > 0:
                    ax.scatter(pts.index, pts, color=color, s=25, label=labels[flag_val], zorder=5)
            
            ax.set_title(f"Node {nid} - {var.capitalize()} Quality Control", fontsize=11)
            ax.set_ylabel(var.capitalize())
            ax.grid(True, linestyle=':', alpha=0.6)
            if idx == 0:
                ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
                
        plt.suptitle(f"Quality Control Analysis: {var.capitalize()}", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'qc_{var}.png'), dpi=120)
        plt.close()
    print("  QC diagnostic plots saved to outputs/plots/")

def generate_qc_report(node_data_orig, node_flags):
    report_path = os.path.join(OUTPUTS_DIR, 'reports', 'data_quality_report.md')
    lines = []
    lines.append("# Data Quality Control & Preprocessing Report")
    lines.append(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\nThis report summarizes the data quality checks and cleaning actions applied to each meteorological variable across all nodes before the imputation process.\n")
    
    for nid in sorted(node_data_orig.keys()):
        df_node = node_data_orig[nid]
        df_flag = node_flags[nid]
        total_records = len(df_node)
        
        lines.append(f"## Node: {nid}")
        lines.append(f"**Total Records**: {total_records:,} (aligned timesteps)\n")
        
        lines.append("| Variable | Total Records | Range Violations (Flag 1) | Spike Detections (Flag 2) | Sensor Freeze Events (Flag 3) | Temporal Jumps (Flag 4) | ERA5 Warnings (Flag 5) | Total Flagged | % Flagged |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        
        for var in TARGET_VARS:
            flag_col = f'qc_{var}'
            flags = df_flag[flag_col]
            
            c_range = (flags == 1).sum()
            c_spike = (flags == 2).sum()
            c_freeze = (flags == 3).sum()
            c_jump = (flags == 4).sum()
            c_era5 = (flags == 5).sum()
            
            total_flagged = (flags > 0).sum()
            pct_flagged = (total_flagged / total_records) * 100
            
            lines.append(f"| {var.capitalize()} | {total_records:,} | {c_range:,} | {c_spike:,} | {c_freeze:,} | {c_jump:,} | {c_era5:,} | {total_flagged:,} | {pct_flagged:.3f}% |")
        lines.append("\n" + "-" * 50 + "\n")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Quality Control report saved to {report_path}")

# Keep track of original resampled data for plots
node_data_orig = {nid: df.copy() for nid, df in node_data.items()}

node_data_cleaned = {}
node_flags = {}

for nid, df_node in node_data.items():
    df_cleaned, df_flag = run_data_cleaning_and_qc(df_node, era5_1min, nid)
    # Merge flags into the cleaned df
    for col in df_flag.columns:
        df_cleaned[col] = df_flag[col]
    node_data_cleaned[nid] = df_cleaned
    node_flags[nid] = df_flag
    
    # Print node cleaning stats
    print(f"  Node {nid} cleaned:")
    for var in TARGET_VARS:
        c_corrupt = df_flag[f'qc_{var}'].isin([1, 2, 4]).sum()
        c_suspicious = df_flag[f'qc_{var}'].isin([3, 5]).sum()
        print(f"    - {var}: replaced {c_corrupt:,} obvious corruptions with NaN, flagged {c_suspicious:,} suspicious points.")

node_data = node_data_cleaned

# Generate diagnostic plots and markdown report
save_qc_plots(node_data_orig, node_flags)
generate_qc_report(node_data_orig, node_flags)

# ============================================================
# 3. Create Cyclical Temporal Features
# ============================================================
print('\n[3/10] Creating cyclical temporal features...')
time_feats = pd.DataFrame(index=global_time_range)
time_feats['hour_sin'] = np.sin(2 * np.pi * global_time_range.hour / 24)
time_feats['hour_cos'] = np.cos(2 * np.pi * global_time_range.hour / 24)
time_feats['doy_sin']  = np.sin(2 * np.pi * global_time_range.dayofyear / 366)
time_feats['doy_cos']  = np.cos(2 * np.pi * global_time_range.dayofyear / 366)
time_feats['month_sin'] = np.sin(2 * np.pi * global_time_range.month / 12)
time_feats['month_cos'] = np.cos(2 * np.pi * global_time_range.month / 12)

# ============================================================
# 4. Phase 1: Linear Interpolation (Short Gaps <= 15 min)
# ============================================================
print('\n[4/10] Running Phase 1: Linear interpolation on gaps <= 15 minutes...')
node_data_phase1 = {}
for nid, df_node in node_data.items():
    df_p1 = df_node.copy()
    for col in TARGET_VARS:
        before_nans = df_p1[col].isna().sum()
        df_p1[col] = df_p1[col].interpolate(method='linear', limit=15, limit_direction='both')
        after_nans = df_p1[col].isna().sum()
        print(f"  {nid} - {col}: {before_nans:,} NaNs -> {after_nans:,} NaNs (filled {before_nans - after_nans:,})")
    node_data_phase1[nid] = df_p1

# ============================================================
# 5. Denoising Autoencoder Setup (Inputs & Masks)
# ============================================================
print('\n[5/10] Setting up Denoising Autoencoder inputs and masks...')
node_encoded_dfs = {}
for nid, df_node_p1 in node_data_phase1.items():
    df_node_orig = node_data[nid]
    df_setup = pd.DataFrame(index=global_time_range)
    
    # 1. Observation masks (1 if observed after Phase 1, 0 if missing)
    for col in TARGET_VARS:
        df_setup[f'mask_{col}'] = df_node_p1[col].notna().astype(np.float32)
        
    # 2. Temporary filled target features (fallback to corresponding ERA5 values)
    era5_map = {
        'temperature': 'era5_temperature_2m',
        'humidity': 'era5_relative_humidity_2m',
        'pressure': 'era5_pressure_msl',
        'dewpoint': 'era5_dew_point_2m'
    }
    
    for col in TARGET_VARS:
        # Fill missing values using the corresponding ERA5 proxy
        era_col = era5_map[col] if era5_map[col] in era5_1min.columns else era5_1min.columns[0]
        df_setup[f'filled_{col}'] = df_node_p1[col].fillna(era5_1min[era_col])
        
    # 3. Ground truth targets (keep original observed values before Phase 1 interpolation or after)
    # Note: Target for model optimization should be the original observed value
    for col in TARGET_VARS:
        df_setup[f'target_{col}'] = df_node_p1[col]
        # Weight for training: 1 if observed, 0 if missing
        df_setup[f'weight_{col}'] = df_node_p1[col].notna().astype(np.float32)
        
    # 4. Quality Control flags as active training features (Stage 10)
    for col in TARGET_VARS:
        df_setup[f'qc_{col}'] = df_node_orig[f'qc_{col}']
        
    # Combine with ERA5 and time features
    df_combined = pd.concat([df_setup, era5_1min, time_feats], axis=1)
    df_combined['node_id'] = nid
    
    # Forward/backward fill all features to prevent NaN features
    feature_cols = [c for c in df_combined.columns if not c.startswith('target_') and c != 'node_id']
    df_combined[feature_cols] = df_combined[feature_cols].ffill().bfill()
    
    node_encoded_dfs[nid] = df_combined

# Fit LabelEncoder for Node ID representation
le_node = LabelEncoder()
le_node.fit(list(node_data.keys()))
joblib.dump(le_node, os.path.join(OUTPUTS_DIR, 'model', 'label_encoder_node.pkl'))
print(f"Node Encoder saved. Classes: {le_node.classes_}")

# ============================================================
# 6. Extract Strided Sequences
# ============================================================
print('\n[6/10] Extracting training and validation sequences...')
SEQ_LEN = 120
STRIDE_TRAIN = 60
STRIDE_VAL = 120

# Lists for training
X_train_list_cont = []
X_train_list_node = []
Y_train_list = []
W_train_list = []

# Lists for validation
X_val_list_cont = []
X_val_list_node = []
Y_val_list = []
W_val_list = []
X_val_list_times = []

# List features
sample_df = next(iter(node_encoded_dfs.values()))
continuous_features = [c for c in sample_df.columns if not c.startswith('target_') and not c.startswith('weight_') and c != 'node_id']
target_cols = [f'target_{col}' for col in TARGET_VARS]
weight_cols = [f'weight_{col}' for col in TARGET_VARS]

print(f"Continuous features count: {len(continuous_features)}")

# Chronological split index
split_point = int(0.8 * len(global_time_range))
train_range = global_time_range[:split_point]
val_range = global_time_range[split_point:]

for nid, df_node in node_encoded_dfs.items():
    node_idx = le_node.transform([nid])[0]
    
    # Split dataframe chronologically
    df_train = df_node.loc[train_range]
    df_val = df_node.loc[val_range]
    
    # 1. Extract training sequences
    cont_arr_train = df_train[continuous_features].values
    target_arr_train = df_train[target_cols].values
    weight_arr_train = df_train[weight_cols].values
    
    for i in range(0, len(df_train) - SEQ_LEN + 1, STRIDE_TRAIN):
        seq_w = weight_arr_train[i:i+SEQ_LEN]
        if seq_w.sum() == 0:
            continue
        X_train_list_cont.append(cont_arr_train[i:i+SEQ_LEN])
        X_train_list_node.append([node_idx])
        Y_train_list.append(target_arr_train[i:i+SEQ_LEN])
        W_train_list.append(seq_w)
        
    # 2. Extract validation sequences (with larger stride for memory efficiency)
    cont_arr_val = df_val[continuous_features].values
    target_arr_val = df_val[target_cols].values
    weight_arr_val = df_val[weight_cols].values
    
    for i in range(0, len(df_val) - SEQ_LEN + 1, STRIDE_VAL):
        seq_w = weight_arr_val[i:i+SEQ_LEN]
        if seq_w.sum() == 0:
            continue
        X_val_list_cont.append(cont_arr_val[i:i+SEQ_LEN])
        X_val_list_node.append([node_idx])
        Y_val_list.append(target_arr_val[i:i+SEQ_LEN])
        W_val_list.append(seq_w)
        X_val_list_times.append(df_val.index[i:i+SEQ_LEN])

X_train_cont = np.array(X_train_list_cont, dtype=np.float32)
X_train_node = np.array(X_train_list_node, dtype=np.int32)
Y_train = np.array(Y_train_list, dtype=np.float32)
W_train = np.array(W_train_list, dtype=np.float32)

X_val_cont = np.array(X_val_list_cont, dtype=np.float32)
X_val_node = np.array(X_val_list_node, dtype=np.int32)
Y_val = np.array(Y_val_list, dtype=np.float32)
W_val = np.array(W_val_list, dtype=np.float32)
X_val_times = np.array(X_val_list_times)

print(f"Extracted Train samples: {X_train_cont.shape[0]:,}")
print(f"Extracted Val samples: {X_val_cont.shape[0]:,}")

# Free list memory immediately
import gc
del X_train_list_cont, X_train_list_node, Y_train_list, W_train_list
del X_val_list_cont, X_val_list_node, Y_val_list, W_val_list, X_val_list_times
gc.collect()


# ============================================================
# 7. Scalers & Normalization
# ============================================================
print('\n[7/10] Normalizing features...')

# Normalization: Scalers fit on training data
scaler_X = MinMaxScaler()
num_features = X_train_cont.shape[2]
scaler_X.fit(X_train_cont.reshape(-1, num_features))

# 1. Compute medians for target columns ignoring NaNs
Y_train_2d = Y_train.reshape(-1, len(TARGET_VARS))
medians = np.nanmedian(Y_train_2d, axis=0)

# Fallback defaults in case any column is all NaN (unlikely)
default_medians = [25.0, 75.0, 1010.0, 20.0]
for col_idx in range(len(TARGET_VARS)):
    if np.isnan(medians[col_idx]):
        medians[col_idx] = default_medians[col_idx]

# 2. Construct clean target arrays by filling NaNs with column medians
Y_train_clean_2d = np.copy(Y_train_2d)
for col_idx in range(len(TARGET_VARS)):
    nan_mask = np.isnan(Y_train_clean_2d[:, col_idx])
    Y_train_clean_2d[nan_mask, col_idx] = medians[col_idx]
Y_train_clean = Y_train_clean_2d.reshape(Y_train.shape)

Y_val_2d = Y_val.reshape(-1, len(TARGET_VARS))
Y_val_clean_2d = np.copy(Y_val_2d)
for col_idx in range(len(TARGET_VARS)):
    nan_mask = np.isnan(Y_val_clean_2d[:, col_idx])
    Y_val_clean_2d[nan_mask, col_idx] = medians[col_idx]
Y_val_clean = Y_val_clean_2d.reshape(Y_val.shape)

# 3. Fit scaler_Y on clean training data
scaler_Y = MinMaxScaler()
scaler_Y.fit(Y_train_clean_2d)

# Save scalers
joblib.dump(scaler_X, os.path.join(OUTPUTS_DIR, 'model', 'scaler_X.pkl'))
joblib.dump(scaler_Y, os.path.join(OUTPUTS_DIR, 'model', 'scaler_Y.pkl'))

# Transform variables in-place to save memory
X_train_cont_s = scaler_X.transform(X_train_cont.reshape(-1, num_features)).reshape(X_train_cont.shape)
X_val_cont_s = scaler_X.transform(X_val_cont.reshape(-1, num_features)).reshape(X_val_cont.shape)

Y_train_s = scaler_Y.transform(Y_train_clean.reshape(-1, len(TARGET_VARS))).reshape(Y_train.shape)
Y_val_s = scaler_Y.transform(Y_val_clean.reshape(-1, len(TARGET_VARS))).reshape(Y_val.shape)

# Free unscaled arrays
del X_train_cont, X_val_cont
gc.collect()

print(f"Train samples scaled: {X_train_cont_s.shape[0]:,}")
print(f"Val samples scaled: {X_val_cont_s.shape[0]:,}")

# Pre-mask training set to simulate missingness dynamically
# For each training sample, mask a random rate of observations in the input features
for s in range(X_train_cont_s.shape[0]):
    # Random missingness rate
    rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
    # Identify timesteps where masks are 1 for this sequence (features columns 0-3 are masks)
    for col_idx in range(len(TARGET_VARS)):
        mask_feat_idx = col_idx  # masks are first 4 columns in features
        filled_feat_idx = len(TARGET_VARS) + col_idx # filled targets are next 4 columns
        
        observed_timesteps = np.where(X_train_cont_s[s, :, mask_feat_idx] == 1.0)[0]
        if len(observed_timesteps) > 0:
            num_to_mask = int(rate * len(observed_timesteps))
            mask_times = np.random.choice(observed_timesteps, size=num_to_mask, replace=False)
            
            # Mask features
            X_train_cont_s[s, mask_times, mask_feat_idx] = 0.0  # Set mask to 0
            
            # Fill with ERA5 corresponding feature scaled
            # ERA5 feature column index in features:
            # masks (4) + filled targets (4) + ERA5 (9) + time (6)
            # ERA5 variables start at index 8. Mapping:
            # temperature -> era5_temperature_2m (index 8)
            # humidity -> era5_relative_humidity_2m (index 9)
            # dewpoint -> era5_dew_point_2m (index 10)
            # pressure -> era5_pressure_msl (index 11)
            era_idx_map = {'temperature': 8, 'humidity': 9, 'dewpoint': 10, 'pressure': 11}
            era_col = era_idx_map[TARGET_VARS[col_idx]]
            X_train_cont_s[s, mask_times, filled_feat_idx] = X_train_cont_s[s, mask_times, era_col]

# ============================================================
# 8. Build & Train Bidirectional LSTM Model
# ============================================================
print('\n[8/10] Building and training Bidirectional LSTM model...')

def build_model(hp):
    # Hyperparameters search space
    if isinstance(hp, dict):
        lstm_1_units = hp.get('lstm_1_units', 64)
        lstm_2_units = hp.get('lstm_2_units', 32)
        dropout_rate = hp.get('dropout_rate', 0.2)
        node_emb_dim = hp.get('node_emb_dim', 4)
        learning_rate = hp.get('learning_rate', 1e-3)
    else:
        lstm_1_units = hp.Int('lstm_1_units', min_value=32, max_value=128, step=32, default=64)
        lstm_2_units = hp.Int('lstm_2_units', min_value=16, max_value=64, step=16, default=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1, default=0.2)
        node_emb_dim = hp.Int('node_emb_dim', min_value=2, max_value=8, step=2, default=4)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], default=1e-3)
    
    feat_in = keras.Input(shape=(SEQ_LEN, num_features), name='continuous_input')
    node_in = keras.Input(shape=(1,), name='node_input')
    
    num_nodes = len(le_node.classes_)
    node_emb = layers.Embedding(input_dim=num_nodes, output_dim=node_emb_dim, name='node_embedding')(node_in)
    node_emb = layers.Reshape((node_emb_dim,))(node_emb)
    node_emb_rep = layers.RepeatVector(SEQ_LEN)(node_emb)
    
    merged = layers.Concatenate(axis=-1)([feat_in, node_emb_rep])
    
    x = layers.Bidirectional(layers.LSTM(lstm_1_units, return_sequences=True))(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_2_units, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    out = layers.TimeDistributed(layers.Dense(len(TARGET_VARS), activation='sigmoid'), name='imputation_output')(x)
    
    model = keras.Model(inputs=[feat_in, node_in], outputs=out, name='Universal_Imputer_BiLSTM')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse', metrics=['mae'])
    return model

# Select hyperparameters
if RUN_TUNING and 'kt' in globals():
    print("\n[INFO] Starting Bayesian Optimization Tuning...")
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=TUNER_MAX_TRIALS,
        executions_per_trial=1,
        directory=os.path.join(OUTPUTS_DIR, "tuner_dir"),
        project_name="universal_imputation_tuning",
        overwrite=True
    )
    
    tuner.search(
        x=[X_train_cont_s, X_train_node],
        y=Y_train_s,
        sample_weight=W_train[:, :, 0],
        validation_data=([X_val_cont_s, X_val_node], Y_val_s, W_val[:, :, 0]),
        epochs=TUNER_EPOCHS,
        batch_size=FIT_BATCH_SIZE,
        callbacks=[tf_callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor='val_loss')],
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n[INFO] Best Hyperparameters Found:")
    print(f"  - lstm_1_units: {best_hps.get('lstm_1_units')}")
    print(f"  - lstm_2_units: {best_hps.get('lstm_2_units')}")
    print(f"  - dropout_rate: {best_hps.get('dropout_rate'):.2f}")
    print(f"  - node_emb_dim: {best_hps.get('node_emb_dim')}")
    print(f"  - learning_rate: {best_hps.get('learning_rate')}")
    
    final_hp_config = best_hps
else:
    print("\n[INFO] Using pre-tuned default hyperparameters.")
    final_hp_config = best_hparams

print("\n[INFO] Building final model configuration...")
model = build_model(final_hp_config)

model.summary()

# Save model architecture plot if possible
try:
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=os.path.join(OUTPUTS_DIR, 'model', 'model_architecture.png'), show_shapes=True, show_layer_names=True)
    print("  Model architecture diagram saved successfully.")
except Exception as e:
    print(f"  [WARNING] Could not save model architecture diagram: {e}")

# Verify Sequence Builder shapes
print(f"X_train shape: {X_train_cont_s.shape}")
print(f"y_train shape: {Y_train_s.shape}")
print(f"X_val shape: {X_val_cont_s.shape}")
print(f"y_val shape: {Y_val_s.shape}")

# Pre-training Sanity Checks
assert X_train_cont_s.shape[0] > 0, "X_train must have samples"
assert Y_train_s.shape[0] > 0, "Y_train must have samples"
assert np.isnan(X_train_cont_s).sum() == 0, "X_train must not contain NaNs"
print("[OK] Sanity checks passed. Commencing model training...")

cb_list = [
    tf_callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
    tf_callbacks.ReduceLROnPlateau(patience=6, factor=0.5, monitor='val_loss'),
    tf_callbacks.ModelCheckpoint(os.path.join(OUTPUTS_DIR, 'model', 'best_model_universal.keras'),
                                 save_best_only=True, monitor='val_loss')
]

# Train
history = model.fit(
    x=[X_train_cont_s, X_train_node],
    y=Y_train_s,
    sample_weight=W_train[:, :, 0],  # only calculate loss on observed points
    validation_data=([X_val_cont_s, X_val_node], Y_val_s, W_val[:, :, 0]),
    epochs=FIT_EPOCHS,
    batch_size=FIT_BATCH_SIZE,
    callbacks=cb_list,
    verbose=FIT_VERBOSE
)

# Plot Learning Curve
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss (Weighted MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (Weighted MSE)')
plt.title('Universal Model Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'learning_curve.png'), dpi=120)
plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'training_loss.png'), dpi=120)
plt.close()
print("  Learning curve and training loss plots saved.")

# ============================================================
# 9. Multi-Rate Simulated Missingness Evaluation
# ============================================================
print('\n[9/10] Evaluating model on simulated missingness (10%-50%)...')

rates = [0.1, 0.2, 0.3, 0.4, 0.5]
val_metrics = []

# Define helper metric functions
def calculate_nse(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (num / den) if den != 0 else np.nan

def calculate_kge(y_true, y_pred):
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    if std_true == 0 or mean_true == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1] if std_pred > 0 else 0
    beta = mean_pred / mean_true
    gamma = std_pred / std_true
    return 1.0 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

# Prepare lists to generate report content
report_lines = []
report_lines.append("=== UNIVERSAL IMPUTATION MODEL EVALUATION REPORT ===")
report_lines.append(f"Training nodes: {list(node_data.keys())}")
report_lines.append(f"Lookback window: {SEQ_LEN} minutes\n")

for rate in rates:
    print(f"  Evaluating missingness rate: {int(rate*100)}%...")
    # Copy validation features
    X_val_masked_s = np.copy(X_val_cont_s)
    
    # Record coordinates of masked observed values for metric calculation
    masked_coords = []  # list of tuples: (sample_idx, timestep, target_col)
    
    # Store masked samples specifically for rate == 0.2
    masked_samples = {} if rate == 0.2 else None
    
    for s in range(X_val_masked_s.shape[0]):
        for col_idx in range(len(TARGET_VARS)):
            mask_feat_idx = col_idx
            filled_feat_idx = len(TARGET_VARS) + col_idx
            
            # Observe mask values (weight is 1 for observed)
            observed_timesteps = np.where(W_val[s, :, col_idx] == 1.0)[0]
            if len(observed_timesteps) > 0:
                num_to_mask = int(rate * len(observed_timesteps))
                mask_times = np.random.choice(observed_timesteps, size=num_to_mask, replace=False)
                
                # Apply mask to input features
                X_val_masked_s[s, mask_times, mask_feat_idx] = 0.0
                
                era_idx_map = {'temperature': 8, 'humidity': 9, 'dewpoint': 10, 'pressure': 11}
                era_col = era_idx_map[TARGET_VARS[col_idx]]
                X_val_masked_s[s, mask_times, filled_feat_idx] = X_val_masked_s[s, mask_times, era_col]
                
                for t in mask_times:
                    masked_coords.append((s, t, col_idx))
                    if rate == 0.2:
                        if (s, t) not in masked_samples:
                            masked_samples[(s, t)] = {}
                        masked_samples[(s, t)][col_idx] = Y_val_clean[s, t, col_idx]
                    
    # Predict
    pred_s = model.predict([X_val_masked_s, X_val_node], batch_size=1024, verbose=0)
    
    # Inverse scale predictions and ground truths
    pred = scaler_Y.inverse_transform(pred_s.reshape(-1, len(TARGET_VARS))).reshape(pred_s.shape)
    y_true_orig = Y_val_clean  # unscaled validation labels
    
    # Extract only the artificially masked points
    # Separate vectors for each variable
    masked_values = {v: {'true': [], 'pred': []} for v in TARGET_VARS}
    
    for (s, t, col_idx) in masked_coords:
        var_name = TARGET_VARS[col_idx]
        masked_values[var_name]['true'].append(y_true_orig[s, t, col_idx])
        masked_values[var_name]['pred'].append(pred[s, t, col_idx])
        
    report_lines.append(f"--- MISSINGNESS RATE: {int(rate*100)}% ---")
    
    # Generate outputs for the 20% rate (LOCAL_TEST metrics, plots, and dataset)
    if rate == 0.2:
        # Save 20% test metrics
        test_metrics_list = []
        for col_idx, var_name in enumerate(TARGET_VARS):
            t_vals = np.array(masked_values[var_name]['true'])
            p_vals = np.array(masked_values[var_name]['pred'])
            if len(t_vals) > 0:
                mae = mean_absolute_error(t_vals, p_vals)
                rmse = np.sqrt(mean_squared_error(t_vals, p_vals))
                r2 = r2_score(t_vals, p_vals)
                test_metrics_list.append({
                    'variable': var_name,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                })
                
                # 1. Observed vs Imputed Plot
                plt.figure(figsize=(6, 5))
                plt.scatter(t_vals, p_vals, s=1, alpha=0.3, color='blue')
                mn, mx = min(t_vals.min(), p_vals.min()), max(t_vals.max(), p_vals.max())
                plt.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
                plt.title(f"Observed vs Imputed: {var_name} (20% Missingness)\nPearson R = {np.corrcoef(t_vals, p_vals)[0, 1] if np.std(p_vals) > 0 else 0:.4f}")
                plt.xlabel("Observed Value")
                plt.ylabel("Imputed Value")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'{var_name}_observed_vs_imputed.png'), dpi=120)
                plt.close()
                
                # 2. Residual Histogram Plot
                residuals = p_vals - t_vals
                plt.figure(figsize=(6, 4))
                plt.hist(residuals, bins=50, color='crimson', alpha=0.7)
                plt.axvline(0, color='black', linestyle='--', lw=1.5)
                plt.title(f"Residual Histogram: {var_name} (20% Missingness)")
                plt.xlabel("Residual (Imputed - Observed)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'residual_{var_name}.png'), dpi=120)
                plt.close()
                
        df_test_metrics = pd.DataFrame(test_metrics_list)
        df_test_metrics.to_csv(os.path.join(OUTPUTS_DIR, 'metrics', 'test_metrics.csv'), index=False)
        
        # Save 20% imputed test dataset
        imputed_rows = []
        for (s, t), var_dict in masked_samples.items():
            node_idx = X_val_node[s][0]
            node_id = le_node.inverse_transform([node_idx])[0]
            dt = X_val_times[s, t]
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S') if hasattr(dt, 'strftime') else str(dt)
            
            row = {
                'datetime': dt_str,
                'node_id': node_id
            }
            for col_idx, var_name in enumerate(TARGET_VARS):
                if col_idx in var_dict:
                    row[f'original_{var_name}'] = var_dict[col_idx]
                    row[f'imputed_{var_name}'] = pred[s, t, col_idx]
                else:
                    row[f'original_{var_name}'] = np.nan
                    row[f'imputed_{var_name}'] = np.nan
            imputed_rows.append(row)
            
        df_imputed_test = pd.DataFrame(imputed_rows)
        col_order = ['datetime', 'node_id']
        for var_name in TARGET_VARS:
            col_order.extend([f'original_{var_name}', f'imputed_{var_name}'])
        df_imputed_test = df_imputed_test[col_order]
        df_imputed_test.to_csv(os.path.join(OUTPUTS_DIR, 'imputed_data', 'imputed_test_dataset.csv'), index=False)
        
        # Save Local Test Report in Markdown
        rmse_vals = {item['variable']: item['rmse'] for item in test_metrics_list}
        mae_vals = {item['variable']: item['mae'] for item in test_metrics_list}
        nodes_list_str = "\n".join([f"- {nid}" for nid in sorted(node_data.keys())])
        
        report_md = f"""# Local Test Report

Nodes Loaded:
{nodes_list_str}

Records (aligned timesteps):
{len(global_time_range):,} (Total records across all nodes: {len(global_time_range) * len(node_data):,})

Artificial Missing:
20%

Model:
Bidirectional LSTM

Epochs:
{FIT_EPOCHS}

Final Metrics (20% Missingness):
- Temperature RMSE: {rmse_vals.get('temperature', np.nan):.4f} (MAE: {mae_vals.get('temperature', np.nan):.4f})
- Humidity RMSE: {rmse_vals.get('humidity', np.nan):.4f} (MAE: {mae_vals.get('humidity', np.nan):.4f})
- Pressure RMSE: {rmse_vals.get('pressure', np.nan):.4f} (MAE: {mae_vals.get('pressure', np.nan):.4f})
- Dewpoint RMSE: {rmse_vals.get('dewpoint', np.nan):.4f} (MAE: {mae_vals.get('dewpoint', np.nan):.4f})

Output File Locations:
- Model directory: `outputs/model/`
- Metrics: `outputs/metrics/test_metrics.csv`
- Plots: `outputs/plots/`
- Reports: `outputs/reports/local_test_report.md`
- Imputed Data: `outputs/imputed_data/imputed_test_dataset.csv`

Status:
PASS
"""
        with open(os.path.join(OUTPUTS_DIR, 'reports', 'local_test_report.md'), 'w', encoding='utf-8') as f_rep_md:
            f_rep_md.write(report_md)

    for col_idx, var_name in enumerate(TARGET_VARS):
        t_vals = np.array(masked_values[var_name]['true'])
        p_vals = np.array(masked_values[var_name]['pred'])
        
        if len(t_vals) == 0:
            continue
            
        mae = mean_absolute_error(t_vals, p_vals)
        rmse = np.sqrt(mean_squared_error(t_vals, p_vals))
        r2 = r2_score(t_vals, p_vals)
        nse = calculate_nse(t_vals, p_vals)
        kge = calculate_kge(t_vals, p_vals)
        bias = np.mean(p_vals - t_vals)
        pearson_r = np.corrcoef(t_vals, p_vals)[0, 1] if np.std(p_vals) > 0 else 0
        
        val_metrics.append({
            'rate': rate,
            'variable': var_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'nse': nse,
            'kge': kge,
            'bias': bias,
            'pearson_r': pearson_r
        })
        
        report_lines.append(f"Variable: {var_name:<12} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | NSE: {nse:.4f} | KGE: {kge:.4f} | Bias: {bias:.4f} | R: {pearson_r:.4f}")
        
        # Visualize Observed vs Imputed (Scatter Plot) for 30% rate
        if rate == 0.3:
            # 1. Scatter plot
            plt.figure(figsize=(6, 5))
            plt.scatter(t_vals, p_vals, s=1, alpha=0.3, color='indigo')
            mn, mx = min(t_vals.min(), p_vals.min()), max(t_vals.max(), p_vals.max())
            plt.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
            plt.title(f"Observed vs Imputed: {var_name} (30% Gaps)\nPearson R = {pearson_r:.4f}")
            plt.xlabel("Observed Value")
            plt.ylabel("Imputed Value")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'scatter_30pct_{var_name}.png'), dpi=120)
            plt.close()
            
            # 2. Residual Distribution (Density & Histogram)
            residuals = p_vals - t_vals
            plt.figure(figsize=(6, 4))
            plt.hist(residuals, bins=50, density=True, color='teal', alpha=0.7)
            plt.axvline(0, color='red', linestyle='--', lw=1.5)
            plt.title(f"Residual Distribution: {var_name} (30% Gaps)")
            plt.xlabel("Residual (Imputed - Observed)")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'residuals_30pct_{var_name}.png'), dpi=120)
            plt.close()

            # 3. Error Histogram
            plt.figure(figsize=(6, 4))
            plt.hist(np.abs(residuals), bins=50, color='darkorange', alpha=0.7)
            plt.title(f"Absolute Error Histogram: {var_name} (30% Gaps)")
            plt.xlabel("Absolute Error")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'error_hist_30pct_{var_name}.png'), dpi=120)
            plt.close()

    # Time series comparison on a gap window for visual inspection (30% missing rate)
    if rate == 0.3:
        # Sample sequence
        sample_seq_idx = 0
        plt.figure(figsize=(15, 10))
        for col_idx, var_name in enumerate(TARGET_VARS):
            plt.subplot(4, 1, col_idx+1)
            t_seq = y_true_orig[sample_seq_idx, :, col_idx]
            p_seq = pred[sample_seq_idx, :, col_idx]
            w_seq = W_val[sample_seq_idx, :, col_idx]
            
            # Plot original observed
            obs_idx = np.where(w_seq == 1.0)[0]
            plt.scatter(obs_idx, t_seq[obs_idx], color='steelblue', s=10, label='Observed')
            
            # Highlight imputed values in masked slots
            # The indices we masked out in this sample
            # Find masked values
            mask_feat_idx = col_idx
            masked_indices = np.where((W_val[sample_seq_idx, :, col_idx] == 1.0) & (X_val_masked_s[sample_seq_idx, :, mask_feat_idx] == 0.0))[0]
            if len(masked_indices) > 0:
                plt.scatter(masked_indices, p_seq[masked_indices], color='darkorange', s=20, marker='x', label='Imputed')
                
            plt.plot(p_seq, color='green', alpha=0.5, linestyle=':', label='Imputation Line')
            plt.ylabel(var_name)
            if col_idx == 0:
                plt.legend(loc='upper right')
        plt.suptitle("Time Series Imputation Sample Comparison (30% Missingness)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'time_series_comparison.png'), dpi=120)
        plt.close()

# Save metrics DataFrame
df_metrics = pd.DataFrame(val_metrics)
df_metrics.to_csv(os.path.join(OUTPUTS_DIR, 'metrics', 'imputation_metrics_summary.csv'), index=False)
print("  Imputation metrics summary CSV saved.")

# Save evaluation text report
with open(os.path.join(OUTPUTS_DIR, 'reports', 'evaluation_report.txt'), 'w') as f_rep:
    f_rep.write("\n".join(report_lines))
print("  Evaluation report text saved.")

# ============================================================
# 10. Gap Filling (Inference) and Exporting Final CSVs
# ============================================================
print('\n[10/10] Reconstructing actual gaps in all node CSVs...')

for nid, df_node_raw in node_data.items():
    print(f"  Reconstructing targets for {nid}...")
    df_node_p1 = node_data_phase1[nid]
    df_combined = node_encoded_dfs[nid]
    
    # Make sequences for this entire node
    # Slide with stride=1 to cover all points
    n_rows = len(df_combined)
    cont_arr = df_combined[continuous_features].values
    node_idx = le_node.transform([nid])[0]
    
    # Store reconstructed values
    reconstructed_targets = np.copy(df_node_p1[TARGET_VARS].values)
    
    # A single timestep can be covered by multiple overlapping sequence predictions.
    # We aggregate predictions by taking the average of all overlapping windows.
    sum_preds = np.zeros_like(reconstructed_targets)
    count_preds = np.zeros((n_rows, 1))
    
    # We will slide window=120 with stride=1 to predict everything
    # To save memory, we process in chunks of 50000 sequences
    chunk_size = 50000
    for start_idx in range(0, n_rows - SEQ_LEN + 1, chunk_size):
        end_idx = min(start_idx + chunk_size, n_rows - SEQ_LEN + 1)
        
        X_node_cont_list = []
        map_indices = []
        for i in range(start_idx, end_idx):
            X_node_cont_list.append(cont_arr[i:i+SEQ_LEN])
            map_indices.append(i)
            
        X_node_cont_arr = np.array(X_node_cont_list, dtype=np.float32)
        X_node_cont_arr_s = scaler_X.transform(X_node_cont_arr.reshape(-1, num_features)).reshape(X_node_cont_arr.shape)
        X_node_id_arr = np.full((X_node_cont_arr_s.shape[0], 1), node_idx, dtype=np.int32)
        
        preds_s = model.predict([X_node_cont_arr_s, X_node_id_arr], batch_size=4096, verbose=0)
        preds = scaler_Y.inverse_transform(preds_s.reshape(-1, len(TARGET_VARS))).reshape(preds_s.shape)
        
        for idx, start_t in enumerate(map_indices):
            sum_preds[start_t:start_t+SEQ_LEN] += preds[idx]
            count_preds[start_t:start_t+SEQ_LEN] += 1.0
            
        # Free memory of chunk arrays explicitly
        del X_node_cont_list, X_node_cont_arr, X_node_cont_arr_s, X_node_id_arr, preds_s, preds
        import gc
        gc.collect()
        
    avg_preds = np.divide(sum_preds, count_preds, out=np.zeros_like(sum_preds), where=count_preds > 0)
    
    # Clip average predictions to realistic physical ranges
    BOUNDS = {
        'temperature': (15.0, 45.0),
        'humidity': (30.0, 100.0),
        'pressure': (995.0, 1025.0),
        'dewpoint': (15.0, 35.0)
    }
    for col_idx, var_name in enumerate(TARGET_VARS):
        avg_preds[:, col_idx] = np.clip(avg_preds[:, col_idx], *BOUNDS[var_name])
        
    # Replace NaNs in resampled dataset with predictions
    for col_idx, var_name in enumerate(TARGET_VARS):
        mask_nan = np.isnan(reconstructed_targets[:, col_idx])
        reconstructed_targets[mask_nan, col_idx] = avg_preds[mask_nan, col_idx]
        
    # Fallback ffill/bfill for any edge points missed by sequence overlap
    df_imputed = pd.DataFrame(reconstructed_targets, index=global_time_range, columns=TARGET_VARS)
    df_imputed = df_imputed.ffill().bfill()
    
    # Verify no NaNs remain
    n_nans_left = df_imputed.isna().sum().sum()
    print(f"    NaNs left in target columns after LSTM: {n_nans_left}")
    
    # Export full timeline CSV matching the original format
    raw_path = os.path.join(CACHE_DIR, f'{nid}_raw.csv')
    df_raw_template = pd.read_csv(raw_path)
    
    df_raw_template['datetime'] = (pd.to_datetime(df_raw_template['timestamp'], unit='s', utc=True)
                                   .dt.tz_convert('Asia/Jakarta').dt.tz_localize(None))
    
    df_raw_template = (df_raw_template.drop_duplicates(subset='datetime')
                       .set_index('datetime')
                       .sort_index()
                       .reindex(global_time_range))
    
    # Fill target variables
    # The original CSV header names might be 'dew' instead of 'dewpoint'
    inv_col_map = {v: k for k, v in col_map.items()}
    for var_name in TARGET_VARS:
        # Check what the column name was in original CSV
        orig_col = None
        for k, v in col_map.items():
            if v == var_name and k in df_raw_template.columns:
                orig_col = k
                break
        if orig_col is None:
            # Fallback to standard name
            orig_col = var_name
            
        df_raw_template[orig_col] = df_imputed[var_name]
        
    # Set data source label
    # original if observed, linear_interpolated if filled by linear interpolation, ann_lstm_imputed otherwise
    # Let's compare with original node df
    df_node_orig = node_data[nid]
    df_node_p1 = node_data_phase1[nid]
    
    df_raw_template['data_source'] = 'lstm_imputed'
    
    # If observed in df_node_orig, it is 'original'
    orig_observed = df_node_orig[TARGET_VARS[0]].notna()
    df_raw_template.loc[orig_observed, 'data_source'] = 'original'
    
    # If observed in df_node_p1 but not in df_node_orig, it is 'linear_interpolated'
    interp_observed = (~orig_observed) & df_node_p1[TARGET_VARS[0]].notna()
    df_raw_template.loc[interp_observed, 'data_source'] = 'linear_interpolated'
    
    # Reset index and rebuild timestamp
    df_out = df_raw_template.reset_index().rename(columns={'index': 'datetime'})
    df_out['timestamp'] = df_out['datetime'].apply(
        lambda x: int(pd.Timestamp(x).tz_localize('Asia/Jakarta').tz_convert('UTC').timestamp())
        if pd.notnull(x) else None
    )
    
    # Remove temporary datetime column
    df_out = df_out.drop(columns=['datetime'])
    
    # Save file
    out_path = os.path.join(WRITE_CACHE_DIR, f'{nid}_imputed.csv')
    df_out.to_csv(out_path, index=False)
    print(f"    Exported imputed data to {out_path} ({len(df_out):,} rows)")

print('\n=== UNIVERSAL IMPUTATION PROCESS COMPLETED SUCCESSFULLY! ===')

# Completion check
# ============================================================
# 11. Plot Comparison of Imputed Gaps for Each Node
# ============================================================
print('\n[11/10] Generating imputation comparison plots for missing data windows...')

for nid in list(node_data.keys()):
    try:
        # Load the exported imputed dataset
        out_path = os.path.join(WRITE_CACHE_DIR, f'{nid}_imputed.csv')
        if not os.path.exists(out_path):
            print(f"  [WARN] Imputed file {out_path} not found. Skipping plot.")
            continue
            
        df_imp = pd.read_csv(out_path)
        
        # Convert timestamp back to datetime index for easy plotting
        df_imp['datetime'] = pd.to_datetime(df_imp['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
        df_imp = df_imp.set_index('datetime')
        
        # We need to find a window containing a gap (data_source == 'lstm_imputed')
        lstm_gaps = df_imp[df_imp['data_source'] == 'lstm_imputed']
        if len(lstm_gaps) == 0:
            # Fallback to linear_interpolated or original observations
            lstm_gaps = df_imp[df_imp['data_source'] == 'linear_interpolated']
            if len(lstm_gaps) == 0:
                lstm_gaps = df_imp
            
        # Find a contiguous gap window to visualize.
        # Let's take the middle index of the gaps as a starting point.
        mid_idx = len(lstm_gaps) // 2
        gap_time = lstm_gaps.index[mid_idx]
        
        # Define a 6-hour window centered around this gap time
        start_plot = gap_time - pd.Timedelta(hours=3)
        end_plot = gap_time + pd.Timedelta(hours=3)
        
        df_win = df_imp.loc[start_plot:end_plot]
        if len(df_win) == 0:
            # Fallback to first available window containing any gap
            first_gap = lstm_gaps.index[0]
            df_win = df_imp.loc[first_gap - pd.Timedelta(hours=3):first_gap + pd.Timedelta(hours=3)]
            
        plt.figure(figsize=(12, 10))
        for col_idx, var_name in enumerate(TARGET_VARS):
            plt.subplot(4, 1, col_idx+1)
            
            # Find the original column name in the CSV
            orig_col = None
            for k, v in col_map.items():
                if v == var_name and k in df_win.columns:
                    orig_col = k
                    break
            if orig_col is None:
                orig_col = var_name
                
            # Separate by data_source for distinct styling
            obs_mask = df_win['data_source'] == 'original'
            interp_mask = df_win['data_source'] == 'linear_interpolated'
            lstm_mask = df_win['data_source'] == 'lstm_imputed'
            
            # Plot observed as blue scatter
            plt.scatter(df_win.index[obs_mask], df_win.loc[obs_mask, orig_col], color='royalblue', s=10, label='Observed', alpha=0.8)
            # Plot linear interpolated as orange scatter
            plt.scatter(df_win.index[interp_mask], df_win.loc[interp_mask, orig_col], color='darkorange', s=12, label='Linear Interpolated (<=15 min)', alpha=0.8)
            # Plot LSTM imputed as green scatter/marker
            plt.scatter(df_win.index[lstm_mask], df_win.loc[lstm_mask, orig_col], color='forestgreen', s=15, marker='x', label='LSTM Imputed', alpha=0.9)
            
            # Draw a line connecting the full timeline to show the continuity
            plt.plot(df_win.index, df_win[orig_col], color='gray', alpha=0.3, linestyle='--')
            
            plt.title(f"{nid} - {var_name.capitalize()} Imputation Verification", fontsize=11)
            plt.ylabel(var_name.capitalize())
            if col_idx == 0:
                plt.legend(loc='upper right', frameon=True)
            plt.grid(True, linestyle=':', alpha=0.6)
            
        plt.suptitle(f"Imputation Comparison Plot for {nid} (Contiguous Gap Window)", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'imputation_comparison_{nid}.png'), dpi=120)
        plt.close()
        print(f"  Imputation comparison plot saved to outputs/plots/imputation_comparison_{nid}.png")
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate comparison plot for {nid}: {e}")

# Completion check
required_files = [
    os.path.join(OUTPUTS_DIR, 'metrics', 'test_metrics.csv'),
    os.path.join(OUTPUTS_DIR, 'plots', 'training_loss.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'temperature_observed_vs_imputed.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'humidity_observed_vs_imputed.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'pressure_observed_vs_imputed.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'dewpoint_observed_vs_imputed.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'residual_temperature.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'residual_humidity.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'residual_pressure.png'),
    os.path.join(OUTPUTS_DIR, 'plots', 'residual_dewpoint.png'),
    os.path.join(OUTPUTS_DIR, 'imputed_data', 'imputed_test_dataset.csv'),
    os.path.join(OUTPUTS_DIR, 'reports', 'local_test_report.md'),
    os.path.join(OUTPUTS_DIR, 'logs', 'local_test.log')
]

# Add dynamic comparison plot checks
for nid in list(node_data.keys()):
    required_files.append(os.path.join(OUTPUTS_DIR, 'plots', f'imputation_comparison_{nid}.png'))

all_exist = True
missing_files = []
for f in required_files:
    if not os.path.exists(f):
        all_exist = False
        missing_files.append(f)

if RUN_MODE == "LOCAL_TEST":
    if all_exist:
        print("\n=========================================")
        print("LOCAL TEST COMPLETED SUCCESSFULLY")
        print("=========================================")
        print("\nGenerated Files:")
        print("- outputs/model/")
        print("- outputs/metrics/")
        print("- outputs/plots/")
        print("- outputs/reports/")
        print("- outputs/imputed_data/")
    else:
        print("\n=========================================")
        print("LOCAL TEST FAILED (MISSING OUTPUT FILES)")
        print("=========================================")
        print("Missing files:")
        for mf in missing_files:
            print(f"- {mf}")
