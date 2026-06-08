import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path

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
try:
    import shap
except ImportError:
    shap = None

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# 1. KONFIGURASI & REPRODUKTIBILITAS (REPRODUCIBILITY)
# ==============================================================================
CONFIG = {
    'SEED': 42,
    'TIME_STEPS_LSTM': 24, # 24 timesteps dari data 3-jaman (72 jam riwayat)
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
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

KELAS_HUJAN = {
    0: "Kering (0 mm)",
    1: "Ringan (0.1-2.5 mm)",
    2: "Sedang (2.5-10 mm)",
    3: "Lebat (>10 mm)"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tetapkan_seed(seed=42):
    """Menjamin reproduktibilitas absolut di numpy, tf, dan python."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Seed acak ditetapkan ke {seed} untuk reproduktibilitas penuh.")

def dapatkan_path():
    """Mendeteksi environment (Kaggle vs Lokal) dan mengembalikan path."""
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
# 2. TABEL TRANSPARANSI DATA (DATASET DOCUMENTATION)
# ==============================================================================
def cetak_tabel_input():
    """Mencetak TABLE 1: INPUT DATA DESCRIPTION."""
    tabel = """
TABLE 1: INPUT DATA DESCRIPTION
| Feature Name | Physical Meaning | Unit | Transformation | Used By Model |
|--------------|------------------|------|----------------|---------------|
| rain_mm | total precipitation | mm | 3-hour rolling sum | Target |
| temperature_era5 | temperature | °C | raw / normalized | Both |
| humidity_era5 | relative humidity | % | raw / normalized | Both |
| dewpoint_depression | temp - dewpoint | °C | calculated feature | Both |
| wind_speed_era5 | wind speed | m/s | raw / normalized | Both |
| wind_u | wind vector u component | - | sin(radians(wind_dir)) | Both |
| wind_v | wind vector v component | - | cos(radians(wind_dir)) | Both |
| pressure_era5 | surface pressure | hPa | raw / normalized | Both |
| pressure_change_*h | pressure tendency | hPa | 1h/3h/6h difference | Both |
| cloudcover_era5 | cloud cover fraction | % | raw / normalized | Both |
| hour_sin/cos | diurnal cycle encoding | - | sine/cosine transform | Both |
| doy_sin/cos | seasonal cycle encoding| - | sine/cosine transform | Both |
| rain_lag_X | hourly rainfall lags | mm | t-1 to t-24 shifts | Both |
| rain_roll_*h | rolling rainfall stats | mm | 3/6/12/24h mean & max | Both |
"""
    logger.info("\\n" + tabel)

def cetak_ringkasan_dataset(n_total, n_train, n_val, n_test, start_date, end_date, n_features):
    """Mencetak TABLE 2: DATASET SUMMARY."""
    tabel = f"""
TABLE 2: DATASET SUMMARY
| Metric | Value |
|--------|-------|
| Total Samples | {n_total} |
| Training Samples | {n_train} |
| Validation Samples | {n_val} |
| Test Samples | {n_test} |
| Date Range | {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} |
| Temporal Resolution | 3-Hourly |
| Number of Features | {n_features} |
"""
    logger.info("\\n" + tabel)

def cetak_distribusi_target(y_train, y_val, y_test):
    """Mencetak TABLE 3: TARGET DISTRIBUTION."""
    y_all = np.concatenate([y_train, y_val, y_test])
    total = len(y_all)
    counts = {i: np.sum(y_all == i) for i in range(4)}
    
    tabel = f"""
TABLE 3: TARGET DISTRIBUTION
| Class | Description | Count | Percentage |
|-------|-------------|-------|------------|
| 0 | Dry (0 mm) | {counts[0]} | {counts[0]/total*100:.2f}% |
| 1 | Light Rain (0.1-2.5 mm) | {counts[1]} | {counts[1]/total*100:.2f}% |
| 2 | Moderate Rain (2.5-10 mm)| {counts[2]} | {counts[2]/total*100:.2f}% |
| 3 | Heavy Rain (>10 mm) | {counts[3]} | {counts[3]/total*100:.2f}% |
"""
    logger.info("\\n" + tabel)

# ==============================================================================
# 3. PEMROSESAN DATA & REKAYASA FITUR (PREPROCESSING & FEATURE ENGINEERING)
# ==============================================================================
def muat_data(filepath):
    logger.info(f"Memuat dataset ERA5 dari {filepath}...")
    if not filepath.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {filepath}")
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
    df = df.sort_index()
    return df

def rekayasa_fitur_dan_validasi(df):
    """Memvalidasi metorologi, mengubah arah angin, dan membuat fitur baru pada data per jam."""
    logger.info("Melakukan validasi meteorologi dan rekayasa fitur...")
    
    # Validasi Meteorologi Dasar
    if (df['rain_mm'] < 0).sum() > 0:
        logger.warning("Ditemukan nilai curah hujan negatif. Memperbaiki ke 0.")
        df.loc[df['rain_mm'] < 0, 'rain_mm'] = 0
        
    if df.isna().sum().sum() > 0:
        logger.warning("Ditemukan nilai NaN (hilang). Melakukan interpolasi linier dan pengisian maju.")
        if 'rain_mm' in df.columns:
            df['rain_mm'] = df['rain_mm'].fillna(0)
        df = df.interpolate(method='linear').bfill().ffill()
        
    # 1. Variabel Sirkular (Arah Angin)
    if 'wind_direction_era5' in df.columns:
        logger.info("Mengubah arah angin (derajat) menjadi vektor wind_u dan wind_v...")
        wind_dir_rad = np.radians(df['wind_direction_era5'])
        df['wind_u'] = np.sin(wind_dir_rad)
        df['wind_v'] = np.cos(wind_dir_rad)
        df = df.drop(columns=['wind_direction_era5'])
        
    # 2. Dewpoint Depression
    if 'temperature_era5' in df.columns and 'dewpoint_era5' in df.columns:
        df['dewpoint_depression'] = df['temperature_era5'] - df['dewpoint_era5']
        
    # 3. Lags Curah Hujan (1-24 Jam)
    for i in range(1, 25):
        df[f'rain_lag_{i}'] = df['rain_mm'].shift(i)
        
    # 4. Rata-rata dan Maksimum Bergerak (Rolling Stats)
    for w in [3, 6, 12, 24]:
        df[f'rain_roll_mean_{w}h'] = df['rain_mm'].rolling(window=w, min_periods=1).mean()
    for w in [3, 6, 12]:
        df[f'rain_roll_max_{w}h'] = df['rain_mm'].rolling(window=w, min_periods=1).max()
        
    # 5. Pressure Tendency (Perubahan Tekanan)
    if 'pressure_era5' in df.columns:
        for w in [1, 3, 6]:
            df[f'pressure_change_{w}h'] = df['pressure_era5'].diff(w)
            
    # 6. Pengkodean Waktu Siklus (Diurnal & Seasonal)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    return df.dropna()

def resampling_ke_3jam(df):
    """Mengagregasi data per jam menjadi blok 3-jaman. Menentukan target kejadian."""
    logger.info("Melakukan resampling data (Agregasi 3 jam)...")
    
    agg_dict = {col: 'last' for col in df.columns}
    agg_dict['rain_mm'] = 'sum'
    
    # Paksa indeks berkelanjutan sebelum agregasi
    idx_harapan = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    df = df.reindex(idx_harapan).ffill()
    df_3h = df.resample('3h').agg(agg_dict).dropna()
    
    def klasifikasi_hujan(mm):
        if mm < 0.1: return 0
        elif mm <= 2.5: return 1
        elif mm <= 10.0: return 2
        else: return 3
        
    df_3h['target_class'] = df_3h['rain_mm'].apply(klasifikasi_hujan)
    return df_3h

def buat_urutan_sekuensial(df_x, y, time_steps):
    """Membuat tensor sekuensial untuk LSTM sambil mempertahankan indeks waktu."""
    Xs, ys, indeks = [], [], []
    for i in range(len(df_x) - time_steps):
        Xs.append(df_x.iloc[i:(i + time_steps)].values)
        ys.append(y[i + time_steps])
        indeks.append(df_x.index[i + time_steps])
    return np.array(Xs), np.array(ys), np.array(indeks)

# ==============================================================================
# 4. PEMBANGUNAN MODEL (MODEL TRAINING)
# ==============================================================================
def latih_xgboost(X_train, y_train):
    """Melatih dan mengkalibrasi model XGBoost Multikelas."""
    logger.info("Melatih model Tabular XGBoost dengan Kalibrasi Isotonik...")
    bobot = class_weight.compute_sample_weight('balanced', y_train)
    
    model_dasar = xgb.XGBClassifier(**CONFIG['XGB_PARAMS'], random_state=CONFIG['SEED'])
    
    # Kalibrasi Probabilitas Isotonik agar akurat secara saintifik
    model_kalibrasi = CalibratedClassifierCV(model_dasar, method='isotonic', cv=3)
    model_kalibrasi.fit(X_train, y_train, sample_weight=bobot)
    return model_kalibrasi

def latih_lstm(X_train, y_train, X_val, y_val, out_dir):
    """Melatih model sekuensial LSTM mendalam dengan TensorFlow Keras."""
    logger.info("Melatih model Sekuensial LSTM...")
    logger.info(f"Dimensi Tensor Input LSTM: {X_train.shape} (Sampel, Timesteps, Fitur)")
    
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax') # Output multi-kelas
    ])
    
    bobot_kelas = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    dict_bobot = dict(enumerate(bobot_kelas))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Keras Callbacks
    jalur_simpan = str(out_dir / "lstm_model.keras")
    panggilan_balik = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(filepath=jalur_simpan, monitor='val_loss', save_best_only=True)
    ]
    
    model.fit(
        X_train, y_train, epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
        validation_data=(X_val, y_val), class_weight=dict_bobot,
        callbacks=panggilan_balik, verbose=0
    )
    return model

# ==============================================================================
# 5. EVALUASI METRIK (Q1 STANDARD EVALUATION)
# ==============================================================================
def expected_calibration_error(y_true_binary, y_prob_binary, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_binary, n_bins=n_bins)
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob_binary, bins) - 1
    bin_total = np.bincount(binids, minlength=len(bins))
    
    ece = 0
    total = len(y_prob_binary)
    for i in range(len(prob_pred)):
        idx = np.abs(bins[:-1] + np.diff(bins)/2 - prob_pred[i]).argmin()
        count = bin_total[idx]
        ece += (count / total) * np.abs(prob_pred[i] - prob_true[i])
    return ece

def evaluasi_ekstrem(y_true, y_pred, indeks_kelas=3):
    """Menghitung metrik ekstrem CSI dan FAR untuk deteksi hujan lebat (>10mm)."""
    benar_ekstrem = (y_true == indeks_kelas).astype(int)
    prediksi_ekstrem = (y_pred == indeks_kelas).astype(int)
    
    hits = np.sum((benar_ekstrem == 1) & (prediksi_ekstrem == 1))
    false_alarms = np.sum((benar_ekstrem == 0) & (prediksi_ekstrem == 1))
    misses = np.sum((benar_ekstrem == 1) & (prediksi_ekstrem == 0))
    
    csi = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else 0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0
    
    return csi, far, recall

def evaluasi_model(y_true, y_prob):
    """
    Menghitung semua metrik Biner dan Multi-kelas yang dipersyaratkan.
    Metode konsisten: Peluang biner hujan diperoleh dari penjumlahan peluang Ringan + Sedang + Lebat.
    """
    y_pred = np.argmax(y_prob, axis=1)
    
    y_true_bin = (y_true > 0).astype(int)
    y_prob_bin = np.clip(y_prob[:, 1:].sum(axis=1), 0.0, 1.0) # Hindari kesalahan presisi floating-point
    y_pred_bin = (y_prob_bin > 0.5).astype(int)
    
    kurva_presisi, kurva_recall, _ = precision_recall_curve(y_true_bin, y_prob_bin)
    pr_auc = auc(kurva_recall, kurva_presisi)
    
    brier = brier_score_loss(y_true_bin, y_prob_bin)
    klimatologi = np.mean(y_true_bin)
    brier_ref = brier_score_loss(y_true_bin, np.full_like(y_prob_bin, klimatologi))
    bss = 1 - (brier / brier_ref) if brier_ref > 0 else 0.0
    
    csi, far, rec_ext = evaluasi_ekstrem(y_true, y_pred, indeks_kelas=3)
    
    metrik = {
        'ROC-AUC': roc_auc_score(y_true_bin, y_prob_bin),
        'PR-AUC': pr_auc,
        'Akurasi Biner': accuracy_score(y_true_bin, y_pred_bin),
        'F1 Biner': f1_score(y_true_bin, y_pred_bin, zero_division=0),
        'Presisi Biner': precision_score(y_true_bin, y_pred_bin, zero_division=0),
        'Recall Biner': recall_score(y_true_bin, y_pred_bin, zero_division=0),
        'Skor Brier': brier,
        'Brier Skill Score': bss,
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Akurasi Multikelas': accuracy_score(y_true, y_pred),
        'Log Loss': log_loss(y_true, y_prob),
        'ECE (Kalibrasi)': expected_calibration_error(y_true_bin, y_prob_bin),
        'Recall Hujan Lebat': rec_ext,
        'CSI (Hujan Lebat)': csi,
        'FAR (Hujan Lebat)': far
    }
    return metrik, y_true_bin, y_prob_bin, y_pred

# ==============================================================================
# 6. VISUALISASI (PLOTTING)
# ==============================================================================
def hasilkan_visualisasi(hasil, plot_dir):
    logger.info("Menghasilkan visualisasi berkualitas publikasi (300 DPI)...")
    sns.set_theme(style="whitegrid")
    
    # 1. Kurva ROC
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, data in hasil.items():
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(data['y_true_bin'], data['y_prob_bin'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {data['metrics']['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Tebakan Acak')
    plt.xlabel('Tingkat Positif Palsu (FPR)')
    plt.ylabel('Tingkat Positif Benar (TPR)')
    plt.title('Kurva ROC (Deteksi Kejadian Hujan)')
    plt.legend()
    plt.savefig(plot_dir / 'kurva_roc.png')
    plt.close()
    
    # 2. Kurva Precision-Recall (Penting untuk Imbalance)
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, data in hasil.items():
        p, r, _ = precision_recall_curve(data['y_true_bin'], data['y_prob_bin'])
        plt.plot(r, p, label=f"{model_name} (PR-AUC = {data['metrics']['PR-AUC']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Kurva Precision-Recall')
    plt.legend()
    plt.savefig(plot_dir / 'kurva_pr.png')
    plt.close()
    
    # 3. Diagram Reliabilitas / Kalibrasi
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, data in hasil.items():
        prob_true, prob_pred = calibration_curve(data['y_true_bin'], data['y_prob_bin'], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{model_name} (ECE = {data['metrics']['ECE (Kalibrasi)']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Kalibrasi Sempurna')
    plt.xlabel('Rata-rata Probabilitas Prediksi')
    plt.ylabel('Pecahan Positif (Kejadian Hujan Aktual)')
    plt.title('Diagram Reliabilitas (Reliability Diagram)')
    plt.legend()
    plt.savefig(plot_dir / 'diagram_reliabilitas.png')
    plt.close()
    
    # 4. Matriks Kebingungan (Confusion Matrix)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    label = [KELAS_HUJAN[i] for i in range(4)]
    for i, (model_name, data) in enumerate(hasil.items()):
        cm = confusion_matrix(data['y_true_multi'], data['y_pred_multi'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label, ax=axes[i])
        axes[i].set_title(f'Matriks Kebingungan {model_name}')
        axes[i].set_xlabel('Prediksi Model')
        axes[i].set_ylabel('Data Aktual')
    plt.tight_layout()
    plt.savefig(plot_dir / 'matriks_kebingungan.png')
    plt.close()
    
    # 5. Fitur Penting (XGBoost SHAP / Gain)
    try:
        model_dasar = hasil['XGBoost']['model'].calibrated_classifiers_[0].estimator
        if shap is not None:
            explainer = shap.TreeExplainer(model_dasar)
            # Batasi hingga 1000 sampel agar cepat
            nilai_shap = explainer.shap_values(hasil['XGBoost']['X_test'][:1000])
            plt.figure(figsize=(10, 6), dpi=300)
            # Menampilkan SHAP untuk kelas Hujan Lebat (indeks 3) jika memungkinkan
            if isinstance(nilai_shap, list):
                shap.summary_plot(nilai_shap[3], hasil['XGBoost']['X_test'][:1000], feature_names=hasil['XGBoost']['nama_fitur'], show=False)
            else:
                shap.summary_plot(nilai_shap, hasil['XGBoost']['X_test'][:1000], feature_names=hasil['XGBoost']['nama_fitur'], show=False)
            plt.title('Nilai Dampak SHAP (SHAP Impact Values) pada Prediksi')
            plt.savefig(plot_dir / 'fitur_penting_shap.png')
            plt.close()
        else:
            raise ImportError("SHAP tidak tersedia")
    except Exception as e:
        logger.warning(f"Melewati plot SHAP ({e}). Mencetak XGBoost Gain Importance standar.")
        try:
            kepentingan = model_dasar.feature_importances_
            idx = np.argsort(kepentingan)[-15:]
            plt.figure(figsize=(10, 6), dpi=300)
            plt.barh(range(len(idx)), kepentingan[idx], align='center')
            plt.yticks(range(len(idx)), hasil['XGBoost']['nama_fitur'][idx])
            plt.title('Fitur Terpenting (Gain) XGBoost - Top 15')
            plt.tight_layout()
            plt.savefig(plot_dir / 'fitur_penting_gain.png')
            plt.close()
        except Exception:
            pass

def cetak_tabel_perbandingan(hasil):
    logger.info("\\n================ PERBANDINGAN MODEL (Q1 Journal Standard) ================")
    metrik_utama = [
        'ROC-AUC', 'PR-AUC', 'Akurasi Biner', 'F1 Biner', 'Skor Brier', 'Brier Skill Score', 
        'Macro F1', 'Log Loss', 'ECE (Kalibrasi)', 'Recall Hujan Lebat', 'CSI (Hujan Lebat)', 'FAR (Hujan Lebat)'
    ]
    
    print("| Metrik | XGBoost | LSTM | Model Terbaik |")
    print("|---|---|---|---|")
    
    for m in metrik_utama:
        xgb_v = hasil['XGBoost']['metrics'][m]
        lstm_v = hasil['LSTM']['metrics'][m]
        
        # Metrik di mana nilai "LEBIH RENDAH lebih baik"
        if m in ['Skor Brier', 'Log Loss', 'ECE (Kalibrasi)', 'FAR (Hujan Lebat)']:
            terbaik = 'XGBoost' if xgb_v < lstm_v else 'LSTM'
        else:
            terbaik = 'XGBoost' if xgb_v > lstm_v else 'LSTM'
            
        print(f"| {m} | {xgb_v:.4f} | {lstm_v:.4f} | **{terbaik}** |")

# ==============================================================================
# 7. EKSEKUSI UTAMA (MAIN PIPELINE)
# ==============================================================================
def main():
    tetapkan_seed(CONFIG['SEED'])
    data_path, out_dir, plot_dir = dapatkan_path()
    
    cetak_tabel_input()
    
    # 1. Pipeline Data
    df = muat_data(data_path)
    # Kami menggunakan data 2017 - 2025 saja (Pemisahan Kronologis yang ketat)
    df = df.loc['2017':'2025'].copy()
    
    df = rekayasa_fitur_dan_validasi(df)
    df_3h = resampling_ke_3jam(df)
    
    fitur_kolom = [c for c in df_3h.columns if c not in ['rain_mm', 'target_class']]
    X_raw = df_3h[fitur_kolom]
    y_raw = df_3h['target_class'].values
    
    # Splitting Kronologis (Menghindari temporal leakage)
    train_mask = df_3h.index < '2024-01-01'
    val_mask = (df_3h.index >= '2024-01-01') & (df_3h.index < '2025-01-01')
    test_mask = df_3h.index >= '2025-01-01'
    
    # Data Bocoran dihentikan: Skalabilitas HANYA pas ke data training!
    scaler = MinMaxScaler()
    scaler.fit(X_raw[train_mask])
    
    # Skalakan semuanya tetapi dipandu oleh MinMaxScaler training
    X_scaled_all = scaler.transform(X_raw)
    df_scaled = pd.DataFrame(X_scaled_all, index=df_3h.index, columns=fitur_kolom)
    
    # Buat Sekuens LSTM menjaga Indeks Timestamp
    X_seq, y_seq, seq_idx = buat_urutan_sekuensial(df_scaled, y_raw, CONFIG['TIME_STEPS_LSTM'])
    
    # Filter Indeks untuk mendapatkan split sequence yang tepat
    seq_train_mask = seq_idx < pd.Timestamp('2024-01-01')
    seq_val_mask = (seq_idx >= pd.Timestamp('2024-01-01')) & (seq_idx < pd.Timestamp('2025-01-01'))
    seq_test_mask = seq_idx >= pd.Timestamp('2025-01-01')
    
    X_train_seq, y_train_seq = X_seq[seq_train_mask], y_seq[seq_train_mask]
    X_val_seq, y_val_seq = X_seq[seq_val_mask], y_seq[seq_val_mask]
    X_test_seq, y_test_seq = X_seq[seq_test_mask], y_seq[seq_test_mask]
    
    # Untuk model Tabular (XGBoost), kita geser target mundur 1 langkah agar data 
    # di akhir window memprediksi 3 jam KEDEPAN.
    X_tab, y_tab = X_scaled_all[:-1], y_raw[1:]
    tab_idx = df_3h.index[1:]
    
    tab_train_mask = tab_idx < pd.Timestamp('2024-01-01')
    tab_val_mask = (tab_idx >= pd.Timestamp('2024-01-01')) & (tab_idx < pd.Timestamp('2025-01-01'))
    tab_test_mask = tab_idx >= pd.Timestamp('2025-01-01')
    
    X_train_xgb, y_train_xgb = X_tab[tab_train_mask], y_tab[tab_train_mask]
    X_test_xgb, y_test_xgb = X_tab[tab_test_mask], y_tab[tab_test_mask]
    
    # Cetak Metadata
    cetak_ringkasan_dataset(
        n_total=len(df_3h),
        n_train=len(X_train_seq),
        n_val=len(X_val_seq),
        n_test=len(X_test_seq),
        start_date=df_3h.index.min(),
        end_date=df_3h.index.max(),
        n_features=len(fitur_kolom)
    )
    cetak_distribusi_target(y_train_seq, y_val_seq, y_test_seq)
    
    # 2. Pelatihan
    xgb_model = latih_xgboost(X_train_xgb, y_train_xgb)
    lstm_model = latih_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, out_dir)
    
    # 3. Pengujian Model
    logger.info("Memprediksi data uji secara probabilistik...")
    xgb_prob = xgb_model.predict_proba(X_test_xgb)
    lstm_prob = lstm_model.predict(X_test_seq)
    
    hasil = {}
    
    m_xgb, y_tb_xgb, y_pb_xgb, y_pred_xgb = evaluasi_model(y_test_xgb, xgb_prob)
    hasil['XGBoost'] = {
        'model': xgb_model, 'nama_fitur': np.array(fitur_kolom), 'X_test': X_test_xgb,
        'metrics': m_xgb, 'y_true_multi': y_test_xgb, 'y_pred_multi': y_pred_xgb,
        'y_true_bin': y_tb_xgb, 'y_prob_bin': y_pb_xgb
    }
    
    m_lstm, y_tb_lstm, y_pb_lstm, y_pred_lstm = evaluasi_model(y_test_seq, lstm_prob)
    hasil['LSTM'] = {
        'metrics': m_lstm, 'y_true_multi': y_test_seq, 'y_pred_multi': y_pred_lstm,
        'y_true_bin': y_tb_lstm, 'y_prob_bin': y_pb_lstm
    }
    
    # 4. Simpan Artifak
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'XGBoost': m_xgb, 'LSTM': m_lstm}, f, indent=4)
        
    df_pred = pd.DataFrame(index=seq_idx[seq_test_mask])
    df_pred['Aktual_Multikelas'] = y_test_seq
    df_pred['LSTM_Prob_Hujan'] = y_pb_lstm
    # Padding array xgb untuk menyelaraskan (XGB punya lebih banyak baris uji daripada LSTM Sequence)
    # Kami menyelaraskan via indeks agar aman.
    df_pred_xgb = pd.DataFrame({'XGB_Prob_Hujan': y_pb_xgb}, index=tab_idx[tab_test_mask])
    df_gabungan = df_pred.join(df_pred_xgb, how='inner')
    df_gabungan.to_csv(out_dir / 'predictions.csv')
    
    hasilkan_visualisasi(hasil, plot_dir)
    cetak_tabel_perbandingan(hasil)
    
    logger.info("Selesai. Pipeline siap dipublikasikan.")

if __name__ == "__main__":
    main()
