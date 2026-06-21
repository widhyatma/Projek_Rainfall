"""
Universal Meteorological Imputation Inference Pipeline
Loads the pre-trained Bidirectional LSTM model, label encoders, and scalers
to run Pre-Imputation QC and gap-filling for a user-specified datetime range and Node ID.
Exports imputed CSV datasets and visual validation plots.
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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# ============================================================
# 1. USER CONFIGURATION PANEL
# ============================================================
# Modify the variables below to set your inference range and target node
START_DATETIME = "2026-06-21 15:23:00"  # Format: YYYY-MM-DD HH:MM:SS
END_DATETIME   = "2026-06-24 23:59:00"  # Format: YYYY-MM-DD HH:MM:SS
NODE_ID        = "id-03"                # Options: id-02, id-03, id-05
RUN_MODE       = "LOCAL"                # LOCAL or KAGGLE env setup

# ============================================================
# 2. Directory & Path Setup
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == 'model_imputasi' else SCRIPT_DIR

if RUN_MODE == "KAGGLE" or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("[INFO] Setting paths for Kaggle Environment")
    CACHE_DIR = '/kaggle/input/notebooks/jerismeteo/cek-data-sensor'
    PATH_ERA5 = '/kaggle/input/notebooks/jerismeteo/generate-era5-data/cuaca_gabungan_jerukagung.csv'
    OUTPUT_BASE_DIR = '/kaggle/working'
    OUTPUTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'outputs')
else:
    print("[INFO] Setting paths for Local Environment")
    CACHE_DIR = os.path.join(BASE_DIR, 'cache_data')
    PATH_ERA5 = os.path.join(BASE_DIR, 'forecast_open_meteo_jerukangung', 'cuaca_gabungan_jerukagung.csv')
    if not os.path.exists(PATH_ERA5):
        PATH_ERA5 = os.path.join(SCRIPT_DIR, 'cuaca_gabungan_jerukagung.csv')
    OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')

os.makedirs(os.path.join(OUTPUTS_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, 'imputed_data'), exist_ok=True)

TARGET_VARS = ['temperature', 'humidity', 'pressure', 'dewpoint']
SEQ_LEN = 120

# Validate user dates
start_ts = pd.to_datetime(START_DATETIME)
end_ts = pd.to_datetime(END_DATETIME)
if start_ts >= end_ts:
    raise ValueError("START_DATETIME must be chronologically before END_DATETIME.")

print(f"\nTarget Range : {start_ts} to {end_ts}")
print(f"Target Node  : {NODE_ID}")

# ============================================================
# 3. Model & Scaler Artifact Loading
# ============================================================
print('\n[1/7] Loading pre-trained model and preprocessing scalers...')
model_dir = os.path.join(OUTPUTS_DIR, 'model')
if not os.path.exists(model_dir):
    # Fallback to local script model output path
    model_dir = os.path.join(SCRIPT_DIR, 'outputs', 'model')

path_model = os.path.join(model_dir, 'best_model_universal.keras')
path_scaler_X = os.path.join(model_dir, 'scaler_X.pkl')
path_scaler_Y = os.path.join(model_dir, 'scaler_Y.pkl')
path_le_node = os.path.join(model_dir, 'label_encoder_node.pkl')

for p in [path_model, path_scaler_X, path_scaler_Y, path_le_node]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required model artifact not found at: {p}. Make sure the training pipeline has been executed first.")

# Load model
try:
    import tensorflow as tf
    from tensorflow import keras
    
    # Hot-patch to handle Keras/Tensorflow serialization version mismatch
    try:
        import keras as keras_standalone
        if hasattr(keras_standalone.layers, 'Embedding'):
            if keras_standalone.layers.Embedding.__init__.__name__ not in ('patched_init', 'patched_init_tf'):
                orig_init = keras_standalone.layers.Embedding.__init__
                def patched_init(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    orig_init(self, *args, **kwargs)
                keras_standalone.layers.Embedding.__init__ = patched_init
    except Exception:
        pass

    try:
        if hasattr(tf.keras.layers, 'Embedding'):
            if tf.keras.layers.Embedding.__init__.__name__ not in ('patched_init', 'patched_init_tf'):
                orig_init_tf = tf.keras.layers.Embedding.__init__
                def patched_init_tf(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    orig_init_tf(self, *args, **kwargs)
                tf.keras.layers.Embedding.__init__ = patched_init_tf
    except Exception:
        pass

    model = keras.models.load_model(path_model)
    print("  Successfully loaded Universal Imputation LSTM model.")
except Exception as e:
    print(f"  [ERROR] Failed to load Keras model: {e}")
    sys.exit(1)

scaler_X = joblib.load(path_scaler_X)
scaler_Y = joblib.load(path_scaler_Y)
le_node = joblib.load(path_le_node)
print("  Successfully loaded labels, scalers, and encoders.")

# ============================================================
# 4. Data Loading with Temporal Padding
# ============================================================
print('\n[2/7] Loading datasets with temporal padding context...')
# Pad target ranges to allow centered lookback/lookahead window prediction
padded_start = start_ts - pd.Timedelta(minutes=SEQ_LEN)
padded_end = end_ts + pd.Timedelta(minutes=SEQ_LEN)
padded_time_range = pd.date_range(start=padded_start, end=padded_end, freq='1min')

# Load node data
node_path = os.path.join(CACHE_DIR, f'{NODE_ID}_raw.csv')
if not os.path.exists(node_path):
    raise FileNotFoundError(f"Stational raw dataset not found at: {node_path}")
df_raw = pd.read_csv(node_path)
df_raw['datetime'] = (pd.to_datetime(df_raw['timestamp'], unit='s', utc=True)
                      .dt.tz_convert('Asia/Jakarta')
                      .dt.tz_localize(None))
df_raw = (df_raw.dropna(subset=['datetime'])
          .sort_values('datetime')
          .drop_duplicates(subset='datetime')
          .set_index('datetime'))

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
df_rename = df_raw.rename(columns=col_map)
df_rename = df_rename.loc[:, ~df_rename.columns.duplicated()]

# Resample to 1-minute mean first to align timestamps to the minute boundary
df_1min = df_rename[TARGET_VARS].resample('1min').mean()
# Reindex stational data to padded range
df_node_padded = df_1min.reindex(padded_time_range)

# Load ERA5 data
if not os.path.exists(PATH_ERA5):
    raise FileNotFoundError(f"ERA5 auxiliary dataset not found at: {PATH_ERA5}")
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
era5_cols = [c for c in era5_req_cols if c in era5_raw.columns]
era5_feats = era5_raw[era5_cols]

# Interpolate ERA5 data to aligned 1min padded index
era5_reindexed = era5_feats.reindex(era5_feats.index.union(padded_time_range)).sort_index()
era5_interpolated = era5_reindexed.interpolate(method='time')
era5_padded = era5_interpolated.reindex(padded_time_range)
era5_padded.columns = [f'era5_{c}' for c in era5_padded.columns]
era5_padded = era5_padded.ffill().bfill()

print(f"  Node aligned size with context padding: {len(df_node_padded):,} rows.")
print(f"  ERA5 features aligned size            : {len(era5_padded):,} rows.")

# ============================================================
# 5. Pre-Imputation Data Cleaning (Meteorological QC)
# ============================================================
print('\n[3/7] Running Meteorological Quality Control (QC) checks...')

def hampel_filter_pandas(series, window=31, n_sigmas=3.0):
    rolling_median = series.rolling(window=window, min_periods=1, center=True).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, min_periods=1, center=True).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (series - rolling_median).abs()
    is_outlier = difference > threshold
    return is_outlier

def detect_sensor_freezes(series, min_minutes=30):
    r_min = series.rolling(window=min_minutes, min_periods=min_minutes).min()
    r_max = series.rolling(window=min_minutes, min_periods=min_minutes).max()
    is_frozen_end = (r_max - r_min) == 0.0
    is_frozen = is_frozen_end[::-1].rolling(window=min_minutes, min_periods=1).max()[::-1].fillna(False).astype(bool)
    return is_frozen

df_flags = pd.DataFrame(0, index=df_node_padded.index, columns=[f'qc_{v}' for v in TARGET_VARS])
df_cleaned = df_node_padded.copy()

era5_map = {
    'temperature': 'era5_temperature_2m',
    'humidity': 'era5_relative_humidity_2m',
    'pressure': 'era5_pressure_msl',
    'dewpoint': 'era5_dew_point_2m'
}
qc_bounds = {
    'temperature': (-10.0, 60.0),
    'humidity': (0.0, 100.0),
    'pressure': (850.0, 1100.0),
    'dewpoint': (-10.0, 60.0)
}
jump_limits = {
    'temperature': 3.0,
    'pressure': 5.0,
    'humidity': 30.0,
    'dewpoint': 3.0
}

for var in TARGET_VARS:
    series = df_node_padded[var]
    flag_col = f'qc_{var}'
    
    # ERA5 Check (Flag 5)
    if var in era5_map and era5_map[var] in era5_padded.columns:
        era_series = era5_padded[era5_map[var]]
        diff_era5 = series - era_series
        mean_bias = np.nanmean(diff_era5)
        std_bias = np.nanstd(diff_era5)
        if std_bias > 0:
            is_era5_inconsistent = (diff_era5 - mean_bias).abs() > 3.0 * std_bias
            df_flags.loc[is_era5_inconsistent, flag_col] = 5
            
    # Freeze Check (Flag 3)
    is_freeze = detect_sensor_freezes(series, min_minutes=30)
    df_flags.loc[is_freeze, flag_col] = 3
    
    # Rate Check (Flag 4)
    diff_temp = series.diff().abs()
    is_jump = diff_temp > jump_limits[var]
    df_flags.loc[is_jump, flag_col] = 4
    
    # Spike Check (Flag 2)
    is_spike = hampel_filter_pandas(series, window=31, n_sigmas=3.0)
    df_flags.loc[is_spike, flag_col] = 2
    
    # Bounds Check (Flag 1)
    vmin, vmax = qc_bounds[var]
    is_out_of_range = (series < vmin) | (series > vmax)
    df_flags.loc[is_out_of_range, flag_col] = 1
    
    # Dewpoint consistency: dewpoint <= temperature
    if var == 'dewpoint' and 'temperature' in df_node_padded.columns:
        is_dew_greater = series > df_node_padded['temperature']
        df_flags.loc[is_dew_greater, flag_col] = 1
        df_flags.loc[is_dew_greater, 'qc_temperature'] = 1
        
    # Clear obvious corruption (1, 2, 4) in the cleaned dataframe
    is_corrupt = df_flags[flag_col].isin([1, 2, 4])
    df_cleaned.loc[is_corrupt, var] = np.nan

    c_corrupt = is_corrupt.sum()
    c_suspicious = df_flags[flag_col].isin([3, 5]).sum()
    print(f"  {var:<12}: flagged {c_corrupt:,} corruptions (NaN-replaced) and {c_suspicious:,} suspicious entries.")

# Save flags back to cleaned df
for col in df_flags.columns:
    df_cleaned[col] = df_flags[col]

# ============================================================
# 6. Interpolation & Feature Engineering
# ============================================================
print('\n[4/7] Engineering features and preparing input scaling...')

# Phase 1: Linear Interpolation for short gaps <= 15 mins
df_p1 = df_cleaned.copy()
for col in TARGET_VARS:
    df_p1[col] = df_p1[col].interpolate(method='linear', limit=15, limit_direction='both')

df_setup = pd.DataFrame(index=padded_time_range)
for col in TARGET_VARS:
    df_setup[f'mask_{col}'] = df_p1[col].notna().astype(np.float32)
    
for col in TARGET_VARS:
    era_col = era5_map[col]
    df_setup[f'filled_{col}'] = df_p1[col].fillna(era5_padded[era_col])


# Cyclical time features
time_feats = pd.DataFrame(index=padded_time_range)
time_feats['hour_sin'] = np.sin(2 * np.pi * padded_time_range.hour / 24)
time_feats['hour_cos'] = np.cos(2 * np.pi * padded_time_range.hour / 24)
time_feats['doy_sin']  = np.sin(2 * np.pi * padded_time_range.dayofyear / 366)
time_feats['doy_cos']  = np.cos(2 * np.pi * padded_time_range.dayofyear / 366)
time_feats['month_sin'] = np.sin(2 * np.pi * padded_time_range.month / 12)
time_feats['month_cos'] = np.cos(2 * np.pi * padded_time_range.month / 12)

df_combined = pd.concat([df_setup, era5_padded, time_feats], axis=1)

# Ensure no NaNs remain in feature columns
feature_cols = [c for c in df_combined.columns]
df_combined[feature_cols] = df_combined[feature_cols].ffill().bfill()

# Extract values and scale
X_cont = df_combined.values
num_features = X_cont.shape[1]
X_cont_scaled = scaler_X.transform(X_cont)

print(f"  Feature columns shape: {X_cont_scaled.shape}")

# ============================================================
# 7. Sliding Window Imputation
# ============================================================
print('\n[5/7] Executing sliding window model predictions...')
N = len(padded_time_range)
node_idx = le_node.transform([NODE_ID])[0]

# Pre-allocate output prediction accumulators
sum_preds = np.zeros((N, len(TARGET_VARS)))
count_preds = np.zeros((N, 1))

# Split inference sequences to slide with stride 1
# To optimize VRAM, compile samples in a batch
X_batch_cont = []
map_indices = []

for i in range(N - SEQ_LEN + 1):
    X_batch_cont.append(X_cont_scaled[i : i+SEQ_LEN])
    map_indices.append(i)

X_batch_arr = np.array(X_batch_cont, dtype=np.float32)
X_batch_node = np.full((X_batch_arr.shape[0], 1), node_idx, dtype=np.int32)

print(f"  Running forward pass for {X_batch_arr.shape[0]:,} overlapping windows...")
preds_s = model.predict([X_batch_arr, X_batch_node], batch_size=4096, verbose=1)
preds = scaler_Y.inverse_transform(preds_s.reshape(-1, len(TARGET_VARS))).reshape(preds_s.shape)

# Aggregate predictions by taking the average of all overlapping windows
for idx, start_t in enumerate(map_indices):
    sum_preds[start_t : start_t+SEQ_LEN] += preds[idx]
    count_preds[start_t : start_t+SEQ_LEN] += 1.0

avg_preds = np.divide(sum_preds, count_preds, out=np.zeros_like(sum_preds), where=count_preds > 0)

# Clip average predictions to physical ranges
BOUNDS = {
    'temperature': (15.0, 45.0),
    'humidity': (30.0, 100.0),
    'pressure': (995.0, 1025.0),
    'dewpoint': (15.0, 35.0)
}
for col_idx, var_name in enumerate(TARGET_VARS):
    avg_preds[:, col_idx] = np.clip(avg_preds[:, col_idx], *BOUNDS[var_name])

# Replace NaNs in resampled target columns with model predictions
reconstructed = np.copy(df_cleaned[TARGET_VARS].values)
for col_idx, var_name in enumerate(TARGET_VARS):
    mask_nan = np.isnan(reconstructed[:, col_idx])
    reconstructed[mask_nan, col_idx] = avg_preds[mask_nan, col_idx]

df_imputed_padded = pd.DataFrame(reconstructed, index=padded_time_range, columns=TARGET_VARS)
df_imputed_padded = df_imputed_padded.ffill().bfill()

# ============================================================
# 8. Un-padding & Dataset Export
# ============================================================
print('\n[6/7] Exporting target time range CSV...')
# Crop dataframe to user's requested range (removes padding)
df_imputed_final = df_imputed_padded.loc[start_ts : end_ts]
df_flags_final = df_flags.loc[start_ts : end_ts]
df_orig_final = df_node_padded.loc[start_ts : end_ts]

# Reconstruct data_source label
# original if observed and QC flagged Good (0) or suspicious (3, 5), imputed otherwise
df_export = df_imputed_final.copy()
df_export['data_source'] = 'lstm_imputed'

for var in TARGET_VARS:
    # If it was observed originally (not NaN) and not replaced with NaN by QC
    was_observed = df_orig_final[var].notna() & (~df_flags_final[f'qc_{var}'].isin([1, 2, 4]))
    df_export.loc[was_observed, 'data_source'] = 'original'

# Reset index and format output columns
df_export = df_export.reset_index().rename(columns={'index': 'datetime'})
df_export['timestamp'] = df_export['datetime'].apply(
    lambda x: int(pd.Timestamp(x).tz_localize('Asia/Jakarta').tz_convert('UTC').timestamp())
)

# Merge QC Flags into output dataset for diagnostics
for var in TARGET_VARS:
    df_export[f'qc_{var}'] = df_flags_final[f'qc_{var}'].values

# Final column sorting
output_cols = ['datetime', 'timestamp'] + TARGET_VARS + [f'qc_{var}' for var in TARGET_VARS] + ['data_source']
df_export = df_export[output_cols]

# Save file
str_start = start_ts.strftime('%Y%m%d_%H%M%S')
str_end = end_ts.strftime('%Y%m%d_%H%M%S')
out_path = os.path.join(OUTPUTS_DIR, 'imputed_data', f'{NODE_ID}_inferred_{str_start}_to_{str_end}.csv')
df_export.to_csv(out_path, index=False)
print(f"  Imputed dataset exported successfully to: {out_path} ({len(df_export):,} rows)")

# ============================================================
# 9. Visualization & Verification Plot
# ============================================================
print('\n[7/7] Generating inference verification plots...')
plot_path = os.path.join(OUTPUTS_DIR, 'plots', f'inference_{NODE_ID}_{str_start}_to_{str_end}.png')

plt.figure(figsize=(12, 10))
for col_idx, var_name in enumerate(TARGET_VARS):
    plt.subplot(len(TARGET_VARS), 1, col_idx + 1)
    
    # Original raw data
    plt.plot(df_orig_final.index, df_orig_final[var_name], color='#424242', label='Original Raw (Observed)', alpha=0.8, linewidth=1.2)
    
    # LSTM Imputed values (where original was NaN or QC flag in 1, 2, 4)
    flag_col = f'qc_{var_name}'
    is_imputed_mask = df_orig_final[var_name].isna() | df_flags_final[flag_col].isin([1, 2, 4])
    
    # Imputed line
    plt.plot(df_imputed_final.index, df_imputed_final[var_name], color='#1e88e5', label='Imputed Timeline (Complete)', alpha=0.9, linewidth=1.2)
    
    # Shade contiguous gap blocks (imputed regions)
    imputed_indices = df_orig_final.index[is_imputed_mask]
    if len(imputed_indices) > 0:
        starts = [imputed_indices[0]]
        ends = []
        for idx in range(1, len(imputed_indices)):
            if (imputed_indices[idx] - imputed_indices[idx-1]) > pd.Timedelta(minutes=1):
                ends.append(imputed_indices[idx-1])
                starts.append(imputed_indices[idx])
        ends.append(imputed_indices[-1])
        
        legend_added = False
        for start, end in zip(starts, ends):
            label = 'Imputed Gap Region' if not legend_added else None
            plt.axvspan(start, end + pd.Timedelta(minutes=1), color='#ffe0b2', alpha=0.6, label=label)
            legend_added = True

    # Scatter points for LSTM Imputations
    imputed_points = df_imputed_final.loc[is_imputed_mask, var_name]
    if len(imputed_points) > 0:
        plt.scatter(imputed_points.index, imputed_points, color='forestgreen', s=10, marker='x', label='LSTM Imputed Gaps', zorder=5)
        
    # Scatter points for QC flagged corruptions
    corrupt_points = df_orig_final.loc[df_flags_final[flag_col].isin([1, 2, 4]), var_name]
    if len(corrupt_points) > 0:
        plt.scatter(corrupt_points.index, corrupt_points, color='red', s=15, marker='o', label='QC Filtered Outliers', zorder=6)
        
    plt.title(f"{NODE_ID} - {var_name.capitalize()} Inference Result", fontsize=11)
    plt.ylabel(var_name.capitalize())
    plt.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.suptitle(f"Imputation Inference Plot: Node {NODE_ID}\nRange: {START_DATETIME} to {END_DATETIME}", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"  Verification plot saved to: {plot_path}")

print('\n=== UNIVERSAL IMPUTATION INFERENCE COMPLETED SUCCESSFULLY! ===')
