import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
import warnings

# Scikit-Learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    log_loss, precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# XGBoost
import xgboost as xgb

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==============================================================================
# 1. KONFIGURASI & REPRODUKSIBILITAS
# ==============================================================================
CONFIG = {
    'SEED': 42,
    'TIME_STEPS_LSTM': 24, # 24 langkah waktu data 3-jaman (72 jam riwayat)
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'TEST_SPLIT': 0.2,
    'VAL_SPLIT': 0.1,
    'XGB_PARAMS': {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
}

CLASSES = {
    0: "Kering (0 mm)",
    1: "Ringan (0.1-2.5 mm)",
    2: "Sedang (2.5-10 mm)",
    3: "Lebat (>10 mm)"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    """Menetapkan seed acak untuk reproduksibilitas absolut."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Semua seed acak diatur ke {seed}")

def get_paths():
    """Mendeteksi environment (Kaggle vs Lokal) dan mengatur path direktori."""
    cwd = Path.cwd()
    if '/kaggle' in str(cwd):
        data_path = Path("/kaggle/input/datasets/jerismeteo/open-meteo-data-kebumen/open_meteo_jerukagung/cuaca_jerukagung.csv")
        out_dir = Path("/kaggle/working/outputs")
    else:
        if cwd.name == "Analisis_Meteorologi":
            data_path = cwd / "open_meteo_jerukagung" / "cuaca_jerukagung.csv"
        else:
            data_path = cwd / "Analisis_Meteorologi" / "open_meteo_jerukagung" / "cuaca_jerukagung.csv"
        out_dir = cwd / "outputs"
        
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    return data_path, out_dir, plot_dir

# ==============================================================================
# 2. VALIDASI DATA METEOROLOGI & DOKUMENTASI
# ==============================================================================
def validate_meteorological_data(df):
    """Melakukan pengecekan konsistensi fisik meteorologi pada dataframe mentah."""
    logger.info("Memvalidasi data meteorologi...")
    
    # Cek Curah Hujan Negatif
    neg_rain = (df['rain_mm'] < 0).sum()
    if neg_rain > 0:
        logger.warning(f"Ditemukan {neg_rain} nilai curah hujan negatif. Disesuaikan ke 0.")
        df.loc[df['rain_mm'] < 0, 'rain_mm'] = 0
        
    # Cek Tekanan Udara Tidak Realistis (Batas < 800 atau > 1100 hPa)
    unrealistic_pressure = ((df['pressure_era5'] < 800) | (df['pressure_era5'] > 1100)).sum()
    if unrealistic_pressure > 0:
        logger.warning(f"Ditemukan {unrealistic_pressure} nilai tekanan atmosfer tidak realistis.")
        
    # Cek Kelembapan (RH > 100 atau < 0)
    unrealistic_rh = ((df['humidity_era5'] < 0) | (df['humidity_era5'] > 100)).sum()
    if unrealistic_rh > 0:
        logger.warning(f"Ditemukan {unrealistic_rh} nilai kelembapan (RH) di luar batas 0-100%. Membatasi ke 0-100.")
        df['humidity_era5'] = np.clip(df['humidity_era5'], 0, 100)
        
    # Cek Missing Timestamps (Interval 1 Jam)
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    missing_times = len(expected_index) - len(df.index)
    if missing_times > 0:
        logger.warning(f"Ditemukan {missing_times} timestamp yang hilang. Interpolasi linear akan diterapkan.")
    
    return df

def calculate_dewpoint_depression(df):
    """Menghitung Dewpoint (Titik Embun) menggunakan formula Magnus-Tetens lalu mencari Dewpoint Depression."""
    # Konstanta Magnus-Tetens
    a = 17.625
    b = 243.04
    alpha = np.log(df['humidity_era5'] / 100.0) + (a * df['temperature_era5']) / (b + df['temperature_era5'])
    dewpoint = (b * alpha) / (a - alpha)
    
    df['dewpoint_depression'] = df['temperature_era5'] - dewpoint
    logger.info("Dewpoint Depression (suhu dikurangi titik embun) berhasil dihitung.")
    return df

def generate_documentation_tables(df_raw, df_3h, feature_cols, train_size, val_size, test_size):
    """Mencetak Tabel 1, Tabel 2, dan Tabel 3 secara otomatis ke konsol."""
    
    table1 = """
================================================================================
TABEL 1: FITUR INPUT (INPUT FEATURES)
| Nama Fitur               | Makna Fisik                | Satuan | Transformasi           | Digunakan |
|--------------------------|----------------------------|--------|------------------------|-----------|
| rain_mm                  | Presipitasi total          | mm     | Sum 3-jam berjalan     | Target    |
| temperature_era5         | Suhu Udara                 | °C     | MinMax Scaler          | Keduanya  |
| humidity_era5            | Kelembapan Relatif         | %      | MinMax Scaler          | Keduanya  |
| dewpoint_depression      | Selisih Suhu & Embun       | °C     | Rumus Magnus           | Keduanya  |
| pressure_era5            | Tekanan Permukaan          | hPa    | MinMax Scaler          | Keduanya  |
| pressure_change_1/3/6h   | Tendensi Tekanan           | hPa    | Selisih Waktu (diff)   | Keduanya  |
| wind_u                   | Vektor Angin (U)           | -      | sin(radian(arah))      | Keduanya  |
| wind_v                   | Vektor Angin (V)           | -      | cos(radian(arah))      | Keduanya  |
| hour_sin / hour_cos      | Siklus Diurnal (Harian)    | -      | Sin/Cos Waktu          | Keduanya  |
| doy_sin / doy_cos        | Siklus Musiman (Tahunan)   | -      | Sin/Cos Hari           | Keduanya  |
| rain_lag_X               | Riwayat Hujan (Lags)       | mm     | T-1 hingga T-24        | Keduanya  |
| rain_roll_mean/max_X     | Statistik Berjalan Hujan   | mm     | Mean/Max Berjalan      | Keduanya  |
================================================================================
"""

    total_samples = len(df_3h)
    start_date = df_3h.index.min().strftime('%Y-%m-%d')
    end_date = df_3h.index.max().strftime('%Y-%m-%d')
    
    table2 = f"""
TABEL 2: RINGKASAN DATASET (DATASET SUMMARY)
| Metrik                   | Nilai                |
|--------------------------|----------------------|
| Total Sampel             | {total_samples}               |
| Sampel Pelatihan (Train) | {train_size}               |
| Sampel Validasi (Val)    | {val_size}               |
| Sampel Pengujian (Test)  | {test_size}               |
| Rentang Waktu (Date)     | {start_date} sd {end_date} |
| Resolusi Waktu           | 3-Jam                |
| Jumlah Fitur (X)         | {len(feature_cols)}               |
================================================================================
"""

    counts = df_3h['target_class'].value_counts().sort_index()
    percentages = (counts / counts.sum()) * 100
    
    table3 = f"""
TABEL 3: DISTRIBUSI TARGET (TARGET DISTRIBUTION)
| Kelas Target               | Jumlah  | Persentase |
|----------------------------|---------|------------|
| Kering (0 mm)              | {counts.get(0,0):<7} | {percentages.get(0,0):.2f}%     |
| Hujan Ringan (0.1-2.5 mm)  | {counts.get(1,0):<7} | {percentages.get(1,0):.2f}%     |
| Hujan Sedang (2.5-10 mm)   | {counts.get(2,0):<7} | {percentages.get(2,0):.2f}%     |
| Hujan Lebat (>10 mm)       | {counts.get(3,0):<7} | {percentages.get(3,0):.2f}%     |
================================================================================
"""
    logger.info("\\n" + table1 + table2 + table3)


# ==============================================================================
# 3. DATA PROCESSING & FEATURE ENGINEERING
# ==============================================================================
def load_data(filepath):
    logger.info(f"Memuat data dari {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {filepath}")
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
    df = df.sort_index()
    return df

def feature_engineering(df):
    """Mengaplikasikan lags, rolling stats, siklus waktu, dan konversi vektor angin."""
    logger.info("Melakukan Rekayasa Fitur (Feature Engineering) per Jam...")
    
    # 1. Menghitung Dewpoint Depression
    df = calculate_dewpoint_depression(df)
    
    # 2. Lags Hujan (1 sampai 24 Jam)
    for i in range(1, 25):
        df[f'rain_lag_{i}'] = df['rain_mm'].shift(i)
        
    # 3. Rolling Statistics Hujan
    for window in [3, 6, 12, 24]:
        df[f'rain_roll_mean_{window}h'] = df['rain_mm'].rolling(window=window, min_periods=1).mean()
    for window in [3, 6, 12]:
        df[f'rain_roll_max_{window}h'] = df['rain_mm'].rolling(window=window, min_periods=1).max()
        
    # 4. Pressure Tendency (Tendensi Tekanan)
    for window in [1, 3, 6]:
        if 'pressure_era5' in df.columns:
            df[f'pressure_change_{window}h'] = df['pressure_era5'].diff(window)
            
    # 5. Encoding Siklus Waktu
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    # 6. Transformasi Vektor Angin (Wajib untuk Akurasi)
    if 'wind_direction_era5' in df.columns:
        logger.info("Mengkonversi arah angin sirkular menjadi vektor wind_u dan wind_v...")
        wind_dir_rad = np.radians(df['wind_direction_era5'])
        df['wind_u'] = np.sin(wind_dir_rad)
        df['wind_v'] = np.cos(wind_dir_rad)
        df = df.drop(columns=['wind_direction_era5'])
    
    return df.dropna()

def preprocess_data(df):
    """Mengeksekusi agregasi ke dalam interval 3-Jam (3-Hourly)."""
    logger.info("Mengagregasi data mentah ke dalam resolusi 3-Jaman...")
    
    # Reindex ke interval jam kontinyu
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    df = df.reindex(expected_index).ffill()
    df = validate_meteorological_data(df)
    
    # Resample
    agg_dict = {col: 'last' for col in df.columns}
    agg_dict['rain_mm'] = 'sum'
    
    df_3h = df.resample('3h').agg(agg_dict).dropna()
    
    # Label Target
    def categorize(mm):
        if mm < 0.1: return 0
        elif mm <= 2.5: return 1
        elif mm <= 10.0: return 2
        else: return 3
        
    df_3h['target_class'] = df_3h['rain_mm'].apply(categorize)
    return df_3h

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# ==============================================================================
# 4. MODEL TRAINING
# ==============================================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    logger.info("Melatih Model XGBoost dengan Kalibrasi Isotonic...")
    weights = class_weight.compute_sample_weight('balanced', y_train)
    
    base_clf = xgb.XGBClassifier(
        **CONFIG['XGB_PARAMS'], 
        random_state=CONFIG['SEED'],
        early_stopping_rounds=15
    )
    
    # Early stopping requires eval_set for the base classifier
    base_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=weights,
        verbose=False
    )
    
    # Kalibrasi
    calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv="prefit")
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf, base_clf

def train_lstm(X_train, y_train, X_val, y_val, out_dir):
    logger.info("Melatih Model Sequence LSTM Keras...")
    
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    chkpt_path = out_dir / "lstm_model.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(chkpt_path), monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]
    
    model.fit(
        X_train, y_train, epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
        validation_data=(X_val, y_val), class_weight=class_weights_dict,
        callbacks=callbacks, verbose=0
    )
    return model

# ==============================================================================
# 5. EVALUATION & METRICS
# ==============================================================================
def expected_calibration_error(y_true_binary, y_prob_binary, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_binary, n_bins=n_bins)
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob_binary, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    bin_total = np.bincount(binids, minlength=len(bins))
    
    ece = 0
    total = len(y_prob_binary)
    for i in range(len(prob_pred)):
        idx = np.abs(bins[:-1] + np.diff(bins)/2 - prob_pred[i]).argmin()
        count = bin_total[idx]
        if total > 0:
            ece += (count / total) * np.abs(prob_pred[i] - prob_true[i])
    return ece

def calculate_extreme_metrics(y_true, y_pred, class_idx=3):
    true_extreme = (y_true == class_idx).astype(int)
    pred_extreme = (y_pred == class_idx).astype(int)
    
    hits = np.sum((true_extreme == 1) & (pred_extreme == 1))
    false_alarms = np.sum((true_extreme == 0) & (pred_extreme == 1))
    misses = np.sum((true_extreme == 1) & (pred_extreme == 0))
    
    csi = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else 0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    
    return csi, far, recall, precision

def evaluate_model(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    
    y_true_bin = (y_true > 0).astype(int)
    y_prob_bin = y_prob[:, 1:].sum(axis=1)
    y_prob_bin = np.clip(y_prob_bin, 0.0, 1.0)
    y_pred_bin = (y_prob_bin > 0.5).astype(int)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_prob_bin)
    pr_auc = auc(recall_curve, precision_curve)
    
    brier = brier_score_loss(y_true_bin, y_prob_bin)
    climatology = np.mean(y_true_bin)
    brier_ref = brier_score_loss(y_true_bin, np.full_like(y_prob_bin, climatology))
    bss = 1 - (brier / brier_ref) if brier_ref > 0 else 0.0
    
    csi, far, rec_ext, prec_ext = calculate_extreme_metrics(y_true, y_pred, class_idx=3)
    
    metrics = {
        'Binary Accuracy': accuracy_score(y_true_bin, y_pred_bin),
        'Binary Precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
        'Binary Recall': recall_score(y_true_bin, y_pred_bin, zero_division=0),
        'Binary F1': f1_score(y_true_bin, y_pred_bin, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_bin, y_prob_bin),
        'PR-AUC': pr_auc,
        'Brier Score': brier,
        'Brier Skill Score': bss,
        'Multi Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Log Loss': log_loss(y_true, y_prob),
        'ECE': expected_calibration_error(y_true_bin, y_prob_bin),
        'Heavy Rain Recall': rec_ext,
        'Heavy Rain Precision': prec_ext,
        'Heavy Rain CSI': csi,
        'Heavy Rain FAR': far
    }
    return metrics, y_true_bin, y_prob_bin, y_pred

# ==============================================================================
# 6. EXPORT & VISUALIZATION
# ==============================================================================
def plot_shap_importance(base_xgb_model, X_test, feature_names, plot_dir):
    try:
        import shap
        logger.info("Memulai evaluasi SHAP untuk interpretasi model XGBoost...")
        
        # XGBoost menggunakan format tabular 2D
        explainer = shap.TreeExplainer(base_xgb_model)
        # Ambil subset agar komputasi SHAP cepat
        shap_values = explainer.shap_values(X_test[:1000])
        
        plt.figure(figsize=(10, 6), dpi=300)
        # shap_values untuk XGBClassifier multi-class adalah list dari array, kita plot untuk kelas hujan berat (3)
        if isinstance(shap_values, list) and len(shap_values) == 4:
            shap.summary_plot(shap_values[3], X_test[:1000], feature_names=feature_names, show=False)
            plt.title("SHAP Summary (Kelas Hujan Lebat)")
        else:
            shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_names, show=False)
            plt.title("SHAP Summary Plot")
            
        plt.tight_layout()
        plt.savefig(plot_dir / 'shap_summary.png', bbox_inches='tight')
        plt.close()
    except ImportError:
        logger.warning("Library 'shap' tidak diinstal. Melewati plot SHAP. Menggunakan Gain Importance bawaan XGBoost.")
    
def export_results(results, df_test_index, plot_dir, out_dir):
    logger.info("Menyimpan visualisasi, pentingnya fitur, dan prediksi...")
    sns.set_theme(style="whitegrid")
    
    # Simpan Prediksi
    df_preds = pd.DataFrame(index=df_test_index)
    df_preds['y_true_bin'] = results['XGBoost']['y_true_bin']
    df_preds['xgb_prob_bin'] = results['XGBoost']['y_prob_bin']
    df_preds['lstm_prob_bin'] = results['LSTM']['y_prob_bin']
    df_preds.to_csv(out_dir / 'predictions.csv')
    
    # Feature Importance (Gain)
    base_model = results['XGBoost']['base_model']
    importances = base_model.feature_importances_
    feat_names = results['XGBoost']['feature_names']
    
    df_imp = pd.DataFrame({'Fitur': feat_names, 'Gain': importances}).sort_values(by='Gain', ascending=False)
    df_imp.to_csv(out_dir / 'feature_importance.csv', index=False)
    
    idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(10, 6), dpi=300)
    plt.barh(range(len(idx)), importances[idx], align='center')
    plt.yticks(range(len(idx)), feat_names[idx])
    plt.title('XGBoost Feature Importance (Top 15 - Berdasarkan Gain)')
    plt.tight_layout()
    plt.savefig(plot_dir / 'xgboost_gain_importance.png')
    plt.close()

    # SHAP Importance
    plot_shap_importance(base_model, results['XGBoost']['X_test_scaled'], feat_names, plot_dir)

    # Reliability Diagram
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, data in results.items():
        prob_true, prob_pred = calibration_curve(data['y_true_bin'], data['y_prob_bin'], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{model_name} (ECE = {data['metrics']['ECE']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Ter-Kalibrasi Sempurna')
    plt.xlabel('Rata-rata Probabilitas Prediksi')
    plt.ylabel('Fraksi Positif Aktual (Hujan)')
    plt.title('Reliability Diagram (Kalibrasi Model)')
    plt.legend()
    plt.savefig(plot_dir / 'calibration_diagrams.png')
    plt.close()

def print_comparison_table(results):
    logger.info("\\n================ PERBANDINGAN MODEL (Standar Jurnal Q1) ================")
    metrics_to_compare = [
        'ROC-AUC', 'PR-AUC', 'Binary Accuracy', 'Binary Precision', 'Binary Recall', 'Binary F1', 
        'Brier Score', 'Brier Skill Score', 'Multi Accuracy', 'Macro F1', 'Weighted F1', 
        'Log Loss', 'ECE', 'Heavy Rain Recall', 'Heavy Rain Precision', 'Heavy Rain CSI', 'Heavy Rain FAR'
    ]
    
    print("| Metrik | XGBoost | LSTM | Model Terbaik |")
    print("|---|---|---|---|")
    
    for m in metrics_to_compare:
        xgb_v = results['XGBoost']['metrics'][m]
        lstm_v = results['LSTM']['metrics'][m]
        
        if m in ['Brier Score', 'Log Loss', 'ECE', 'Heavy Rain FAR']:
            best = 'XGBoost' if xgb_v < lstm_v else 'LSTM'
        else:
            best = 'XGBoost' if xgb_v > lstm_v else 'LSTM'
            
        print(f"| {m} | {xgb_v:.4f} | {lstm_v:.4f} | **{best}** |")

# ==============================================================================
# 7. ALUR UTAMA (MAIN)
# ==============================================================================
def main():
    seed_everything(CONFIG['SEED'])
    data_path, out_dir, plot_dir = get_paths()
    
    # 1. Pemuatan & Rekayasa Fitur
    df_raw = load_data(data_path)
    df_eng = feature_engineering(df_raw)
    df_3h = preprocess_data(df_eng)
    
    # Batas analisis mulai dari 2017
    df_3h = df_3h.loc['2017':].copy()
    feature_cols = [c for c in df_3h.columns if c not in ['rain_mm', 'target_class']]
    
    # 2. Pembagian Time-Series Chronological Tanpa Leakage
    n_total = len(df_3h)
    train_end = int(n_total * (1 - CONFIG['TEST_SPLIT']))
    val_end = int(train_end * (1 - CONFIG['VAL_SPLIT'])) # Validasi diambil dari akhir data training
    
    # Kami memastikan scaler hanya fit pada TRAIN SET (Anti-Leakage!)
    scaler = MinMaxScaler()
    scaler.fit(df_3h.iloc[:train_end][feature_cols].values)
    
    # Skalakan seluruh data untuk mempermudah sequence slicing
    X_scaled_all = scaler.transform(df_3h[feature_cols].values)
    y_all = df_3h['target_class'].values
    
    # Setup Data XGBoost (Predict N+1)
    X_xgb, y_xgb = X_scaled_all[:-1], y_all[1:]
    
    # Penyesuaian batas agar target test persis dimulai dari y_all[train_end]
    train_end_xgb = train_end - 1
    val_end_xgb = val_end - 1
    
    X_train_xgb = X_xgb[:train_end_xgb]
    y_train_xgb = y_xgb[:train_end_xgb]
    
    X_val_xgb = X_train_xgb[val_end_xgb:]
    y_val_xgb = y_train_xgb[val_end_xgb:]
    
    X_train_xgb_fit = X_train_xgb[:val_end_xgb]
    y_train_xgb_fit = y_train_xgb[:val_end_xgb]
    
    X_test_xgb = X_xgb[train_end_xgb:]
    y_test_xgb = y_xgb[train_end_xgb:]
    
    # Setup Data LSTM (Sequence Predict)
    # Kami membutuhkan 24 jam ke belakang, jadi indeks pengujian (test_start) bergeser sedikit
    X_seq, y_seq = create_sequences(X_scaled_all, y_all, CONFIG['TIME_STEPS_LSTM'])
    seq_train_end = train_end - CONFIG['TIME_STEPS_LSTM']
    seq_val_end = val_end - CONFIG['TIME_STEPS_LSTM']
    
    X_train_seq = X_seq[:seq_val_end]
    y_train_seq = y_seq[:seq_val_end]
    X_val_seq = X_seq[seq_val_end:seq_train_end]
    y_val_seq = y_seq[seq_val_end:seq_train_end]
    X_test_seq = X_seq[seq_train_end:]
    y_test_seq = y_seq[seq_train_end:]
    
    generate_documentation_tables(
        df_raw, df_3h, feature_cols, 
        train_size=train_end, val_size=(train_end - val_end), test_size=(n_total - train_end)
    )
    
    # 3. Model Training
    calib_xgb, base_xgb = train_xgboost(X_train_xgb_fit, y_train_xgb_fit, X_val_xgb, y_val_xgb)
    lstm_model = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, out_dir)
    
    # 4. Evaluasi
    logger.info("Mengevaluasi kinerja model pada Data Pengujian (Test Set)...")
    xgb_prob = calib_xgb.predict_proba(X_test_xgb)
    lstm_prob = lstm_model.predict(X_test_seq)
    
    results = {}
    
    m_xgb, y_tb_xgb, y_pb_xgb, y_pred_xgb = evaluate_model(y_test_xgb, xgb_prob)
    results['XGBoost'] = {
        'base_model': base_xgb, 'calib_model': calib_xgb, 'feature_names': np.array(feature_cols),
        'X_test_scaled': X_test_xgb, 'metrics': m_xgb, 'y_true_multi': y_test_xgb, 'y_pred_multi': y_pred_xgb,
        'y_true_bin': y_tb_xgb, 'y_prob_bin': y_pb_xgb
    }
    
    m_lstm, y_tb_lstm, y_pb_lstm, y_pred_lstm = evaluate_model(y_test_seq, lstm_prob)
    results['LSTM'] = {
        'metrics': m_lstm, 'y_true_multi': y_test_seq, 'y_pred_multi': y_pred_lstm,
        'y_true_bin': y_tb_lstm, 'y_prob_bin': y_pb_lstm
    }
    
    # 5. Metadata JSON
    metadata = {
        'dates': {'start': str(df_3h.index.min()), 'end': str(df_3h.index.max())},
        'samples': {'train': train_end, 'test': n_total - train_end},
        'features': feature_cols
    }
    with open(out_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    with open(out_dir / 'metrics.json', 'w') as f:
        safe_metrics = {'XGBoost': m_xgb, 'LSTM': m_lstm}
        json.dump(safe_metrics, f, indent=4)
        
    # Visualisasi
    test_dates = df_3h.index[train_end:] 
    export_results(results, test_dates, plot_dir, out_dir)
    
    print_comparison_table(results)
    logger.info("Eksekusi Pipeline Selesai Secara Sukses.")

if __name__ == "__main__":
    main()
