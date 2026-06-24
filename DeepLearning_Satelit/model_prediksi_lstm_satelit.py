"""
Two-Stage LSTM Satellite-Enhanced Rainfall Prediction Pipeline
Trains a Two-Stage forecasting framework (Stage 1: Rain Occurrence, Stage 2: Rain Amount)
by incorporating satellite variables from ERA5 Land, GSMaP, IMERG, and Oya datasets.
Configurable for local and Kaggle environments with LOCAL_TEST and FULL_TRAIN modes.
"""
import os
import sys
import random
import warnings
import logging
import json
import glob
from pathlib import Path

# Data and ML libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script runs
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

# Scikit-Learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, log_loss, mean_squared_error, 
    mean_absolute_error, r2_score, precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import calibration_curve

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# 1. RUN CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================
RUN_MODE = "LOCAL_TEST"  # Options: "LOCAL_TEST" or "FULL_TRAIN"

# Seed configuration
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(SEED)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Starting Satellite-Enhanced LSTM Rainfall Prediction Pipeline in {RUN_MODE} mode.")

# Dynamically set training parameters based on RUN_MODE
if RUN_MODE == "LOCAL_TEST":
    EPOCHS_OCC = 2
    EPOCHS_REG = 2
    OPTUNA_TRIALS = 1
    BATCH_SIZE = 64
else:
    EPOCHS_OCC = 15
    EPOCHS_REG = 20
    OPTUNA_TRIALS = 5  # Can be increased for longer training
    BATCH_SIZE = 64

# Keras Embedding Layer Hot-Patch to prevent serialization issues
try:
    import keras as keras_standalone
    if hasattr(keras_standalone.layers, 'Embedding'):
        orig_init = keras_standalone.layers.Embedding.__init__
        def patched_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig_init(self, *args, **kwargs)
        keras_standalone.layers.Embedding.__init__ = patched_init
except Exception:
    pass

try:
    if hasattr(tf.keras.layers, 'Embedding'):
        orig_init_tf = tf.keras.layers.Embedding.__init__
        def patched_init_tf(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig_init_tf(self, *args, **kwargs)
        tf.keras.layers.Embedding.__init__ = patched_init_tf
except Exception:
    pass

# Dynamic path resolution
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

if IS_KAGGLE:
    logger.info("Kaggle environment detected.")
    OUTPUTS_DIR = Path('/kaggle/working/outputs')
else:
    logger.info("Local environment detected.")
    OUTPUTS_DIR = Path(os.path.join(SCRIPT_DIR, 'outputs'))

# Create directories
os.makedirs(OUTPUTS_DIR / 'plots', exist_ok=True)
os.makedirs(OUTPUTS_DIR / 'metrics', exist_ok=True)
os.makedirs(OUTPUTS_DIR / 'model', exist_ok=True)

def find_file(filename, default_local_path):
    if IS_KAGGLE:
        matches = glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
        if matches:
            return matches[0]
        # Direct fallback guess
        fallback_map = {
            'cuaca_jerukagung.csv': '/kaggle/input/datasets/jerismeteo/open-meteo-data-kebumen/open_meteo_jerukagung/cuaca_jerukagung.csv',
            'ERA5_Land_Standard_Units_TimeSeries_UTC_WMO.csv': '/kaggle/input/ERA5_Land_Standard_Units_TimeSeries_UTC_WMO.csv',
            'Rainfall_GSMaP_TimeSeries_UNIX.csv': '/kaggle/input/Rainfall_GSMaP_TimeSeries_UNIX.csv',
            'Rainfall_IMERG_TimeSeries_UNIX.csv': '/kaggle/input/Rainfall_IMERG_TimeSeries_UNIX.csv',
            'Rainfall_Oya_TimeSeries_UNIX.csv': '/kaggle/input/Rainfall_Oya_TimeSeries_UNIX.csv'
        }
        if filename in fallback_map and os.path.exists(fallback_map[filename]):
            return fallback_map[filename]
    return default_local_path

# Resolve paths
stational_path = find_file('cuaca_jerukagung.csv', os.path.join(BASE_DIR, 'Analisis_Meteorologi', 'open_meteo_jerukagung', 'cuaca_jerukagung.csv'))
era5_path = find_file('ERA5_Land_Standard_Units_TimeSeries_UTC_WMO.csv', os.path.join(BASE_DIR, 'Analisis_Meteorologi', 'Data_Satelit', 'ERA5_Land_Standard_Units_TimeSeries_UTC_WMO.csv'))
gsmap_path = find_file('Rainfall_GSMaP_TimeSeries_UNIX.csv', os.path.join(BASE_DIR, 'Analisis_Meteorologi', 'Data_Satelit', 'Rainfall_GSMaP_TimeSeries_UNIX.csv'))
imerg_path = find_file('Rainfall_IMERG_TimeSeries_UNIX.csv', os.path.join(BASE_DIR, 'Analisis_Meteorologi', 'Data_Satelit', 'Rainfall_IMERG_TimeSeries_UNIX.csv'))
oya_path = find_file('Rainfall_Oya_TimeSeries_UNIX.csv', os.path.join(BASE_DIR, 'Analisis_Meteorologi', 'Data_Satelit', 'Rainfall_Oya_TimeSeries_UNIX.csv'))

for name, p in [('Stational', stational_path), ('ERA5 Land', era5_path), ('GSMaP', gsmap_path), ('IMERG', imerg_path), ('Oya', oya_path)]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required dataset not found for {name} at: {p}")
    logger.info(f"Resolved path for {name}: {p}")

# ==============================================================================
# 2. DATA INGESTION, TIME ALIGNMENT, & MERGING
# ==============================================================================
logger.info("Ingesting and aligning stational and satellite datasets...")

# Helper to read satellite data and localize UTC to WIB (Asia/Jakarta) timezone
def load_satellite_file(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime_utc'], utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    df = df.set_index('datetime').sort_index()
    return df

# A. ERA5 Land
logger.info("Processing ERA5 Land...")
df_era5 = load_satellite_file(era5_path).drop(columns=['unixtime', 'datetime_utc'], errors='ignore')
df_era5.columns = [f'sat_era5_{c}' for c in df_era5.columns]

# B. GSMaP
logger.info("Processing GSMaP...")
df_gsmap = load_satellite_file(gsmap_path).drop(columns=['unixtime', 'datetime_utc'], errors='ignore')
df_gsmap.columns = [f'sat_gsmap_{c}' for c in df_gsmap.columns]

# C. IMERG (30-min to hourly by sum)
logger.info("Processing IMERG...")
df_imerg_raw = load_satellite_file(imerg_path)
df_imerg = df_imerg_raw.drop(columns=['unixtime', 'datetime_utc'], errors='ignore').resample('1h').sum()
df_imerg.columns = [f'sat_imerg_{c}' for c in df_imerg.columns]

# D. Oya (30-min to hourly by sum)
logger.info("Processing Oya...")
df_oya_raw = load_satellite_file(oya_path)
df_oya = df_oya_raw.drop(columns=['unixtime', 'datetime_utc'], errors='ignore').resample('1h').sum()
df_oya.columns = [f'sat_oya_{c}' for c in df_oya.columns]

# Merge satellite data
logger.info("Merging satellite datasets...")
sat_combined = df_era5.join([df_gsmap, df_imerg, df_oya], how='outer')
sat_combined = sat_combined.interpolate(method='linear').ffill().bfill()

# E. Stational
logger.info("Processing Stational Data...")
df_stational = pd.read_csv(stational_path)
df_stational['datetime'] = pd.to_datetime(df_stational['datetime'], utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
df_stational = df_stational.set_index('datetime').sort_index()

essential_cols = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'rain', 
    'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'surface_pressure', 
    'sunshine_duration', 'shortwave_radiation', 'wet_bulb_temperature_2m', 'vapour_pressure_deficit'
]
cols_to_keep = [c for c in essential_cols if c in df_stational.columns]
df_stational = df_stational[cols_to_keep]
if 'rain' in df_stational.columns:
    df_stational.loc[df_stational['rain'] < 0, 'rain'] = 0

# F. Final Join (Keep only valid data after Oya's start date aligned at 2005-01-01)
logger.info("Aligning stational and satellite datasets...")
df_merged = df_stational.join(sat_combined, how='inner')
df_merged = df_merged.loc['2005-01-01 00:00:00':]
df_merged = df_merged.interpolate(method='linear').ffill().bfill()
logger.info(f"Aligned dataset size: {df_merged.shape[0]:,} rows.")

# ==============================================================================
# 3. REKAYASA FITUR (SIMPLIFIED FOR LSTM) & RESAMPLING 3-JAM
# ==============================================================================
# Filosofi: LSTM menangkap pola temporal melalui lookback sequence.
# Lag eksplisit dan rolling statistics per kolom TIDAK DIPERLUKAN
# karena LSTM sudah memiliki mekanisme memori temporal sendiri.
# Target: ~25-30 fitur yang secara fisika informatif (vs 212 sebelumnya).
# ==============================================================================
logger.info("Rekayasa fitur: pendekatan minimalis untuk LSTM...")

# Kumpulkan semua kolom baru dengan pd.concat (hindari PerformanceWarning)
derived_cols = {}

# ---------------------------------------------------------------
# A. Dekomposisi Vektor Angin (Stasional)
#    Menggantikan wind_speed + wind_direction dengan komponen U/V
# ---------------------------------------------------------------
if 'wind_speed_10m' in df_merged.columns and 'wind_direction_10m' in df_merged.columns:
    wd_rad = df_merged['wind_direction_10m'] * np.pi / 180.0
    derived_cols['wind_u'] = -df_merged['wind_speed_10m'] * np.sin(wd_rad)
    derived_cols['wind_v'] = -df_merged['wind_speed_10m'] * np.cos(wd_rad)

# ---------------------------------------------------------------
# B. Indikator Fisika Atmosfer
# ---------------------------------------------------------------
# B1. Dew Point Depression: indikator jenuh udara (suhu - titik embun)
if 'temperature_2m' in df_merged.columns and 'dew_point_2m' in df_merged.columns:
    derived_cols['dewpoint_depression'] = df_merged['temperature_2m'] - df_merged['dew_point_2m']

# B2. Pressure Change 3h: tren tekanan udara, sangat prediktif untuk hujan
if 'surface_pressure' in df_merged.columns:
    derived_cols['pressure_change_3h'] = df_merged['surface_pressure'].diff(3)

# ---------------------------------------------------------------
# C. Konsensus Curah Hujan Satelit
#    Mencerminkan tingkat kesepakatan antar produk satelit
# ---------------------------------------------------------------
sat_rain_raw_cols = [
    'sat_gsmap_hourlyPrecipRate',
    'sat_imerg_precipitation_mmhr',
    'sat_oya_precipitation_mmhr',
    'sat_era5_total_precipitation_hourly_mm'
]
sat_rain_available = [c for c in sat_rain_raw_cols if c in df_merged.columns]

if sat_rain_available:
    sat_rain_df = df_merged[sat_rain_available]
    derived_cols['sat_rain_mean'] = sat_rain_df.mean(axis=1)      # Rata-rata antar produk
    derived_cols['sat_rain_max']  = sat_rain_df.max(axis=1)       # Maksimum (deteksi extreme)
    derived_cols['sat_rain_std']  = sat_rain_df.std(axis=1).fillna(0)  # Ketidaksepakatan

# ---------------------------------------------------------------
# D. ERA5 Temperature Delta (selisih suhu satelit vs stasional)
# ---------------------------------------------------------------
if 'sat_era5_temperature_2m_C' in df_merged.columns and 'temperature_2m' in df_merged.columns:
    derived_cols['era5_temp_delta'] = df_merged['sat_era5_temperature_2m_C'] - df_merged['temperature_2m']

# ---------------------------------------------------------------
# E. Fitur Siklik Waktu (pola harian dan musiman)
# ---------------------------------------------------------------
derived_cols['sin_hour']  = np.sin(2 * np.pi * df_merged.index.hour / 24.0)
derived_cols['cos_hour']  = np.cos(2 * np.pi * df_merged.index.hour / 24.0)
derived_cols['sin_month'] = np.sin(2 * np.pi * df_merged.index.month / 12.0)
derived_cols['cos_month'] = np.cos(2 * np.pi * df_merged.index.month / 12.0)

# ---------------------------------------------------------------
# F. Pilih kolom stasional inti (informatif & tidak redundan)
# ---------------------------------------------------------------
stational_core = [
    'rain',                     # Curah hujan stasional (target & fitur konteks)
    'temperature_2m',           # Suhu udara
    'relative_humidity_2m',     # Kelembaban relatif
    'surface_pressure',         # Tekanan udara
    'vapour_pressure_deficit',  # Defisit tekanan uap
    'sunshine_duration',        # Durasi sinar matahari
    'shortwave_radiation',      # Radiasi gelombang pendek
]
stational_cols = [c for c in stational_core if c in df_merged.columns]

# Pilih kolom ERA5 atmosfer (non-curah-hujan, non-redundan)
era5_atm_core = [
    'sat_era5_u_component_of_wind_10m_ms',   # Komponen U angin ERA5
    'sat_era5_v_component_of_wind_10m_ms',   # Komponen V angin ERA5
    'sat_era5_surface_pressure_hPa',          # Tekanan permukaan ERA5
]
era5_atm_cols = [c for c in era5_atm_core if c in df_merged.columns]

# ---------------------------------------------------------------
# G. Gabungkan semua kolom dengan pd.concat (performa optimal)
# ---------------------------------------------------------------
df_base    = df_merged[stational_cols + sat_rain_available + era5_atm_cols].copy()
df_derived = pd.DataFrame(derived_cols, index=df_merged.index)
df_feat    = pd.concat([df_base, df_derived], axis=1)
df_feat    = df_feat.dropna()

# Log breakdown fitur
n_total = df_feat.shape[1]
logger.info(f"Breakdown fitur (LSTM-optimized):")
logger.info(f"  Stasional inti     : {len(stational_cols)} fitur")
logger.info(f"  Satelit curah hujan: {len(sat_rain_available)} fitur")
logger.info(f"  ERA5 atmosfer      : {len(era5_atm_cols)} fitur")
logger.info(f"  Turunan/derived    : {len(derived_cols)} fitur")
logger.info(f"  TOTAL              : {n_total} fitur (reduksi dari 212)")
logger.info(f"  CATATAN: Lookback temporal ditangani LSTM via sequence_length, bukan lag.")

# ---------------------------------------------------------------
# H. Resampling 3-jam
# ---------------------------------------------------------------
logger.info("Resampling ke interval 3-jam...")
agg_rules = {}
for col in df_feat.columns:
    if 'rain' in col.lower() or 'precip' in col.lower() or 'precipitation' in col.lower():
        agg_rules[col] = 'sum'
    else:
        agg_rules[col] = 'mean'

df_3h = df_feat.resample('3h').agg(agg_rules).dropna()

# Buat target prediksi
df_3h['target_amount']     = df_3h['rain'].shift(-1)
df_3h = df_3h.dropna()
df_3h['target_occurrence'] = (df_3h['target_amount'] >= 1.0).astype(int)  # Hujan ≥1mm/3jam

occ_rate = df_3h['target_occurrence'].mean()
logger.info(f"Total sampel 3-jam : {df_3h.shape[0]:,}")
logger.info(f"Jumlah fitur akhir : {df_3h.shape[1] - 2}")
logger.info(f"Tingkat kejadian   : {occ_rate:.1%} (threshold ≥1mm/3jam)")

# ==============================================================================
# 4. CHRONOLOGICAL SPLIT & SCALING
# ==============================================================================
logger.info("Splitting dataset chronologically...")
train_mask = (df_3h.index.year >= 2005) & (df_3h.index.year <= 2023)
val_mask = (df_3h.index.year == 2024)
test_mask = (df_3h.index.year == 2025)

# Fallback split if years are empty
if df_3h[train_mask].empty:
    train_mask = df_3h.index < df_3h.index[int(len(df_3h) * 0.7)]
    val_mask = (df_3h.index >= df_3h.index[int(len(df_3h) * 0.7)]) & (df_3h.index < df_3h.index[int(len(df_3h) * 0.85)])
    test_mask = df_3h.index >= df_3h.index[int(len(df_3h) * 0.85)]

X = df_3h.drop(columns=['target_amount', 'target_occurrence'])
y = df_3h[['target_amount', 'target_occurrence']]
feature_names = X.columns.tolist()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Sliced regression subsets (amount predicting trained ONLY on rainy instances target_occurrence == 1)
train_rain_mask = y_train['target_occurrence'] == 1
val_rain_mask = y_val['target_occurrence'] == 1
test_rain_mask = y_test['target_occurrence'] == 1

X_train_reg, y_train_reg = X_train[train_rain_mask], y_train[train_rain_mask]
X_val_reg, y_val_reg = X_val[val_rain_mask], y_val[val_rain_mask]
X_test_reg, y_test_reg = X_test[test_rain_mask], y_test[test_rain_mask]

logger.info(f"Train samples (Occ): {X_train.shape[0]:,}, Train samples (Reg, Rain > 0): {X_train_reg.shape[0]:,}")

# Fit scaler
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

X_train_reg_s = scaler.transform(X_train_reg)
X_val_reg_s = scaler.transform(X_val_reg)
X_test_reg_s = scaler.transform(X_test_reg)

# Save the fitted scaler
joblib.dump(scaler, OUTPUTS_DIR / 'model' / 'scaler_prediction.pkl')
logger.info("Scaler exported successfully.")

# ==============================================================================
# 5. METRICS & PLOTTING FUNCTIONS
# ==============================================================================
def met_metrics(y_true, y_pred):
    hits = np.sum((y_pred == 1) & (y_true == 1))
    misses = np.sum((y_pred == 0) & (y_true == 1))
    false_alarms = np.sum((y_pred == 1) & (y_true == 0))
    correct_negatives = np.sum((y_pred == 0) & (y_true == 0))
    
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    
    total = hits + misses + false_alarms + correct_negatives
    hits_random = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else 0
    ets = (hits - hits_random) / (hits + misses + false_alarms - hits_random) if (hits + misses + false_alarms - hits_random) > 0 else 0
    hss = (2 * (hits * correct_negatives - misses * false_alarms)) / ((hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (false_alarms + correct_negatives)) if total > 0 else 0
    
    return {'CSI': float(csi), 'POD': float(pod), 'FAR': float(far), 'ETS': float(ets), 'HSS': float(hss)}

def evaluate_models(y_test_occ, prob_occ_uncal, prob_occ_cal, pred_occ_cal, y_test_reg, pred_reg):
    rmse = float(np.sqrt(mean_squared_error(y_test_reg, pred_reg)))
    mae = float(mean_absolute_error(y_test_reg, pred_reg))
    r2 = float(r2_score(y_test_reg, pred_reg))
    nse = float(1 - (np.sum((y_test_reg - pred_reg)**2) / (np.sum((y_test_reg - np.mean(y_test_reg))**2) + 1e-5)))
    r_pearson = float(np.corrcoef(y_test_reg, pred_reg)[0,1])
    kge = float(1 - np.sqrt((r_pearson - 1)**2 + (np.std(pred_reg)/(np.std(y_test_reg)+1e-5) - 1)**2 + (np.mean(pred_reg)/(np.mean(y_test_reg)+1e-5) - 1)**2))
    
    met = met_metrics(y_test_occ, pred_occ_cal)
    
    report = {
        'Regression_RainyDaysOnly': {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse, 'KGE': kge},
        'Classification': {
            'Accuracy': float(accuracy_score(y_test_occ, pred_occ_cal)),
            'Precision': float(precision_score(y_test_occ, pred_occ_cal, zero_division=0)),
            'Recall': float(recall_score(y_test_occ, pred_occ_cal, zero_division=0)),
            'F1': float(f1_score(y_test_occ, pred_occ_cal, zero_division=0)),
            'ROC_AUC': float(roc_auc_score(y_test_occ, prob_occ_cal)),
            'Brier_Uncalibrated': float(brier_score_loss(y_test_occ, prob_occ_uncal)),
            'Brier_Calibrated': float(brier_score_loss(y_test_occ, prob_occ_cal))
        },
        'Meteorological': met
    }
    return report

def plot_calibration(y_true, prob_uncal, prob_iso, prob_platt, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    f_op_uncal, m_pv_uncal = calibration_curve(y_true, prob_uncal, n_bins=10)
    f_op_iso, m_pv_iso = calibration_curve(y_true, prob_iso, n_bins=10)
    f_op_platt, m_pv_platt = calibration_curve(y_true, prob_platt, n_bins=10)
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(m_pv_uncal, f_op_uncal, "s-", alpha=0.5, label="Uncalibrated", color='orange')
    plt.plot(m_pv_iso, f_op_iso, "s-", label="Isotonic", color='blue')
    plt.plot(m_pv_platt, f_op_platt, "s-", label="Platt Scaling", color='green')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    sns.histplot(prob_uncal, color='orange', alpha=0.2, label='Uncal', kde=True, bins=15)
    sns.histplot(prob_iso, color='blue', alpha=0.2, label='Isotonic', kde=True, bins=15)
    sns.histplot(prob_platt, color='green', alpha=0.2, label='Platt', kde=True, bins=15)
    plt.legend()
    plt.title("Probability Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def plot_classification_diagnostics(y_true, pred, prob, save_path):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(confusion_matrix(y_true, pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.3f}', color='darkorange', linewidth=2)
    plt.plot([0,1],[0,1],'k--', color='gray')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    pr, rc, _ = precision_recall_curve(y_true, prob)
    plt.plot(rc, pr, label=f'AUC-PR={auc(rc, pr):.3f}', color='forestgreen', linewidth=2)
    plt.legend(loc='lower left')
    plt.title('Precision-Recall')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def plot_regression_diagnostics(y_true, pred, save_path):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, pred, alpha=0.4, color='royalblue', edgecolors='k', s=20)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', linewidth=2)
    plt.xlabel('Observed (mm)')
    plt.ylabel('Predicted (mm)')
    plt.title('Pred vs Obs (Rainy Only)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    residuals = y_true - pred
    sns.histplot(residuals, kde=True, color='crimson', bins=20)
    plt.title('Residual Distribution')
    plt.xlabel('Residual (mm)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(y_true[:80], label='Observed', color='black', alpha=0.7)
    plt.plot(pred[:80], label='Predicted', color='dodgerblue', alpha=0.9, linestyle='--')
    plt.legend()
    plt.title('Time Series Snippet')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

# ==============================================================================
# 6. BAYESIAN OPTUNA SEQUENTIAL TUNING
# ==============================================================================
def create_seq(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def obj_lstm_occ(trial):
    """Objective Optuna untuk Stage 1 (Klasifikasi Kejadian Hujan)."""
    keras.backend.clear_session()
    # Dengan 26 fitur (vs 212), memory footprint sequence kecil → bisa explore lookback lebih panjang
    # 8 = 24 jam, 16 = 48 jam, 24 = 72 jam (3 hari), 36 = 108 jam, 48 = 144 jam (6 hari)
    ts      = trial.suggest_categorical('sequence_length', [8, 16, 24, 36, 48])
    hidden  = trial.suggest_int('lstm_units', 32, 128, step=32)
    dropout = trial.suggest_float('dropout_rate', 0.1, 0.4)
    lr      = trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True)
    n_layers = trial.suggest_categorical('n_lstm_layers', [1, 2])

    Xt, yt = create_seq(X_train_s, y_train['target_occurrence'], ts)
    Xv, yv = create_seq(X_val_s,   y_val['target_occurrence'],   ts)
    if len(Xt) < 10 or len(Xv) < 5:
        return 1.0

    inp = keras.Input(shape=(ts, Xt.shape[2]))
    x = inp
    if n_layers == 2:
        x = layers.LSTM(hidden, return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy')
    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    history = model.fit(Xt, yt, validation_data=(Xv, yv),
                        epochs=EPOCHS_OCC, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)
    return min(history.history['val_loss'])

# Run Optuna Study
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger.info(f"Optuna Bayesian Optimization (Stage 1 Classifier, {OPTUNA_TRIALS} trials)...")
logger.info(f"Search space: sequence_length=[8,16,24,36,48], lstm_units=[32,64,96,128], n_layers=[1,2]")
study_lstm_occ = optuna.create_study(direction='minimize')
study_lstm_occ.optimize(obj_lstm_occ, n_trials=OPTUNA_TRIALS)
p_lstm_occ = study_lstm_occ.best_params
p_lstm_reg = p_lstm_occ.copy()  # Stage 2 menggunakan hyperparameter yang sama

logger.info(f"Optuna Best Parameters (Stage 1): {p_lstm_occ}")
logger.info(f"  → Lookback terpilih: {p_lstm_occ['sequence_length']} steps = {p_lstm_occ['sequence_length']*3} jam")

# ==============================================================================
# 7. MODEL TRAINING & CALIBRATION (STAGE 1 & 2)
# ==============================================================================
ts = p_lstm_occ['sequence_length']

# A. Rebuild sequences using optimal window length
Xt_o, yt_o = create_seq(X_train_s, y_train['target_occurrence'], ts)
Xv_o, yv_o = create_seq(X_val_s, y_val['target_occurrence'], ts)
Xte_o, yte_o = create_seq(X_test_s, y_test['target_occurrence'], ts)

Xt_r, yt_r = create_seq(X_train_reg_s, y_train_reg['target_amount'], ts)
Xv_r, yv_r = create_seq(X_val_reg_s, y_val_reg['target_amount'], ts)
Xte_r, yte_r = create_seq(X_test_reg_s, y_test_reg['target_amount'], ts)

# Keras Callbacks
cb_occ = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
    keras.callbacks.ModelCheckpoint(str(OUTPUTS_DIR / 'model' / 'best_lstm_occ_satelit.keras'), monitor='val_loss', save_best_only=True)
]

cb_reg = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0),
    keras.callbacks.ModelCheckpoint(str(OUTPUTS_DIR / 'model' / 'best_lstm_reg_satelit.keras'), monitor='val_loss', save_best_only=True)
]

# 1. Train Occurrence Model (Stage 1 Classifier)
# Arsitektur LSTM dengan Dropout + BatchNormalization untuk regularisasi
logger.info("Training Stage 1 Classifier LSTM (Rain Occurrence)...")
keras.backend.clear_session()

n_layers = p_lstm_occ.get('n_lstm_layers', 1)
hidden   = p_lstm_occ['lstm_units']
dropout  = p_lstm_occ.get('dropout_rate', 0.2)

inp = keras.Input(shape=(ts, Xt_o.shape[2]))
x = inp
if n_layers == 2:
    x = layers.LSTM(hidden, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
x = layers.LSTM(hidden)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout)(x)
out = layers.Dense(1, activation='sigmoid')(x)
clf = keras.Model(inp, out)
clf.compile(optimizer=keras.optimizers.Adam(p_lstm_occ['learning_rate']), loss='binary_crossentropy')
h_occ = clf.fit(Xt_o, yt_o, validation_data=(Xv_o, yv_o),
                epochs=EPOCHS_OCC, batch_size=BATCH_SIZE, verbose=1, callbacks=cb_occ)

# Get predictions
val_prob_uncal  = clf.predict(Xv_o).flatten()
test_prob_uncal = clf.predict(Xte_o).flatten()

# 2. Probability Calibration
logger.info("Performing probability calibration...")
iso_cal = IsotonicRegression(out_of_bounds='clip')
iso_cal.fit(val_prob_uncal, yv_o)
val_prob_iso = iso_cal.predict(val_prob_uncal)
brier_iso = brier_score_loss(yv_o, val_prob_iso)

# Guard: Platt hanya digunakan jika val set punya 2 kelas
if len(np.unique(yv_o)) >= 2:
    platt_cal = LogisticRegression()
    platt_cal.fit(val_prob_uncal.reshape(-1, 1), yv_o)
    val_prob_platt = platt_cal.predict_proba(val_prob_uncal.reshape(-1, 1))[:, 1]
    brier_platt = brier_score_loss(yv_o, val_prob_platt)
else:
    logger.warning("Val set hanya punya 1 kelas — Platt Scaling dilewati, gunakan Isotonic.")
    brier_platt = brier_iso + 1.0  # pastikan Isotonic dipilih
    platt_cal = iso_cal  # fallback
    val_prob_platt = val_prob_iso

logger.info(f"Isotonic Brier Score: {brier_iso:.5f} | Platt Brier Score: {brier_platt:.5f}")

if brier_iso <= brier_platt:
    logger.info("Selecting Isotonic Calibration.")
    selected_calibrator = "Isotonic"
    test_prob_cal = iso_cal.predict(test_prob_uncal)
    joblib.dump(iso_cal, OUTPUTS_DIR / 'model' / 'calibrator.pkl')
else:
    logger.info("Selecting Platt Calibration.")
    selected_calibrator = "Platt"
    test_prob_cal = platt_cal.predict_proba(test_prob_uncal.reshape(-1, 1))[:, 1]
    joblib.dump(platt_cal, OUTPUTS_DIR / 'model' / 'calibrator.pkl')

test_pred_cal   = (test_prob_cal >= 0.5).astype(int)
test_prob_iso   = iso_cal.predict(test_prob_uncal)
test_prob_platt = platt_cal.predict_proba(test_prob_uncal.reshape(-1, 1))[:, 1] if hasattr(platt_cal, 'predict_proba') and platt_cal is not iso_cal else test_prob_iso

# 3. Train Rain Amount Model (Stage 2 Regressor)
# Menggunakan hyperparameter yang sama dengan Stage 1
logger.info("Training Stage 2 Regressor LSTM (Rain Amount)...")
keras.backend.clear_session()

n_layers_reg = p_lstm_reg.get('n_lstm_layers', 1)
hidden_reg   = p_lstm_reg['lstm_units']
dropout_reg  = p_lstm_reg.get('dropout_rate', 0.2)

inp_r = keras.Input(shape=(ts, Xt_r.shape[2]))
x_r = inp_r
if n_layers_reg == 2:
    x_r = layers.LSTM(hidden_reg, return_sequences=True)(x_r)
    x_r = layers.Dropout(dropout_reg)(x_r)
x_r = layers.LSTM(hidden_reg)(x_r)
x_r = layers.BatchNormalization()(x_r)
x_r = layers.Dropout(dropout_reg)(x_r)
out_r = layers.Dense(1, activation='linear')(x_r)
reg = keras.Model(inp_r, out_r)
reg.compile(optimizer=keras.optimizers.Adam(p_lstm_reg['learning_rate']), loss=keras.losses.Huber())
h_reg = reg.fit(Xt_r, yt_r, validation_data=(Xv_r, yv_r),
                epochs=EPOCHS_REG, batch_size=BATCH_SIZE, verbose=1, callbacks=cb_reg)

# Predict regression
test_pred_reg = np.maximum(0, reg.predict(Xte_r).flatten())

# ==============================================================================
# 8. PIPELINE EVALUATION AND METRIC EXPORT
# ==============================================================================
logger.info("Evaluating models and generating plots...")
report = evaluate_models(yte_o, test_prob_uncal, test_prob_cal, test_pred_cal, yte_r, test_pred_reg)

report['Selected_Calibrator'] = selected_calibrator
report['Optuna_Parameters']   = p_lstm_occ
report['N_Features']          = len(feature_names)
report['Feature_Names']       = feature_names
report['Lookback_Steps']      = ts
report['Lookback_Hours']      = ts * 3

# Save metrics report
with open(OUTPUTS_DIR / 'metrics' / 'report.json', 'w') as f:
    json.dump(report, f, indent=4)
logger.info(f"Metrics Report exported to: {OUTPUTS_DIR / 'metrics' / 'report.json'}")
logger.info(f"Fitur: {len(feature_names)} | Lookback: {ts} steps ({ts*3} jam)")

# Generate diagrams and plots
plot_calibration(yte_o, test_prob_uncal, test_prob_iso, test_prob_platt, OUTPUTS_DIR / 'plots' / 'calibration_curves.png')
plot_classification_diagnostics(yte_o, test_pred_cal, test_prob_cal, OUTPUTS_DIR / 'plots' / 'classification_diagnostics.png')
plot_regression_diagnostics(yte_r, test_pred_reg, OUTPUTS_DIR / 'plots' / 'regression_residuals.png')

# Save training history loss curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h_occ.history['loss'], label='Train')
plt.plot(h_occ.history['val_loss'], label='Val')
plt.title('Stage 1 (Classifier) Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h_reg.history['loss'], label='Train')
plt.plot(h_reg.history['val_loss'], label='Val')
plt.title('Stage 2 (Regressor) Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'plots' / 'training_loss_curves.png', dpi=120)
plt.close()

logger.info("\n=== SATELLITE-ENHANCED RAINFOREST PIPELINE COMPLETED SUCCESSFULLY! ===")
print(json.dumps(report, indent=4))
