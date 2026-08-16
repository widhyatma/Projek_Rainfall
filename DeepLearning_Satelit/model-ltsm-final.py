#!/usr/bin/env python
# coding: utf-8

# KUMPULAN IMPORT LIBRARY (DIRESTRUKTURISASI & DIDEDUPLIKASI DI AWAL SKRIP)

import gc
import glob
import joblib
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import sys
import tensorflow as tf
import warnings
from IPython.display import Image, display
from optuna.pruners import MedianPruner
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import backend as K, layers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# --- LSTM - PRE-TRAINING IKLIM (ERA5) Interval 1 Jam ---

# --- Tahap 1: Inisialisasi dan Impor Library ---
# Sel 1: Impor Library dan Setup Lingkungan
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['lines.linewidth'] = 2.0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info("Kerangka Pemodelan Prediksi Curah Hujan 1-Jam Dimulai...")
# Fungsi utilitas untuk mengecek proporsi dataset secara terpisah
def print_dataset_distribution(title, X_tr, y_tr, X_va, y_va, X_te, y_te):
    n_train = len(X_tr)
    n_val = len(X_va)
    n_test = len(X_te)
    total = n_train + n_val + n_test
    y_tr_np = np.array(y_tr)
    y_va_np = np.array(y_va)
    y_te_np = np.array(y_te)
    rain = int(np.sum(y_tr_np == 1) + np.sum(y_va_np == 1) + np.sum(y_te_np == 1))
    non_rain = int(np.sum(y_tr_np == 0) + np.sum(y_va_np == 0) + np.sum(y_te_np == 0))
    rain_pct = (rain / total) * 100 if total > 0 else 0
    non_rain_pct = (non_rain / total) * 100 if total > 0 else 0
    pos_neg_ratio = rain / (non_rain + 1e-9)
    print("")
    print(f"Dataset Distribution ({title})")
    print("")
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"Testing samples: {n_test}")
    print(f"Rain samples: {rain}")
    print(f"Non-rain samples: {non_rain}")
    print(f"Rain percentage: {rain_pct:.2f}%")
    print(f"Non-rain percentage: {non_rain_pct:.2f}%")
    print(f"Positive/Negative ratio: {pos_neg_ratio:.4f}")
    print("=" * 30)
def cek_proporsi_pretraining(X_tr, y_tr, X_va, y_va, X_te, y_te):
    print_dataset_distribution("Pretraining", X_tr, y_tr, X_va, y_va, X_te, y_te)
def cek_proporsi_finetuning(X_tr, y_tr, X_va, y_va, X_te, y_te):
    print_dataset_distribution("Fine-Tuning", X_tr, y_tr, X_va, y_va, X_te, y_te)

# --- Tahap 2: Konfigurasi Global & Konstanta ---
# Sel 2: Konfigurasi Global
# File ini mendefinisikan konstanta fisik, rentang waktu pre-training, dan parameter HPO LSTM.
# Seluruh pengaturan bias correction telah dinonaktifkan/dihapus agar model fokus pada machine learning murni.
RUN_MODE = "FULL_TRAIN"  # "FULL_TRAIN" untuk latihan penuh, "PIPELINE_TEST" untuk uji coba cepat
SEED = 281225
if RUN_MODE == "PIPELINE_TEST":
    OPTUNA_TRIALS    = 5
    EPOCHS_HPO       = 2   # Epoch per trial Optuna (singkat agar eksplorasi cepat)
    EPOCHS_BEST      = 2   # Epoch training model terbaik (lebih panjang, biarkan EarlyStopping bekerja)
    EPOCHS_FINETUNE  = 2   # Epoch fine-tuning pada data AWS
else:
    OPTUNA_TRIALS    = 50  # Jumlah trial optimasi hyperparameter dengan Optuna
    EPOCHS_HPO       = 30  # Epoch per trial Optuna — singkat, eksplorasi cepat, EarlyStopping handle sisanya
    EPOCHS_BEST      = 60 # Epoch training model terbaik — biarkan EarlyStopping + ReduceLR konvergensi penuh
    EPOCHS_FINETUNE  = 60  # Epoch fine-tuning — lebih konservatif agar tidak overfit data AWS
# Panjang urutan waktu (sequence) ke belakang untuk LSTM
LOOKBACK = 24
# Direktori output model dan visualisasi
BASE_OUTPUT_DIR = Path('outputs_lstm')
# Batas ambang curah hujan untuk klasifikasi hujan/tidak hujan (mm/jam)
RAIN_THRESHOLD = 0.2
# Rentang tanggal pre-training menggunakan data satelit & ERA5
PRETRAIN__TRAIN_START = '2005-01-01'
PRETRAIN__TRAIN_END   = '2020-12-31'
PRETRAIN__VAL_START   = '2021-01-01'
PRETRAIN__VAL_END     = '2023-12-31'
PRETRAIN__TEST_START  = '2024-01-01'
PRETRAIN__TEST_END    = '2025-12-31'
# Batas tanggal fine-tuning menggunakan data aktual stasiun AWS bumi
FINETUNE_TRAIN_START = '2025-01-01'
FINETUNE_TRAIN_END   = '2026-01-15'
FINETUNE_VAL_START   = '2026-01-16'
FINETUNE_VAL_END     = '2026-04-05'
FINETUNE_TEST_START  = '2026-04-06'
FINETUNE_TEST_END    = '2026-05-31'
# Pemetaan fitur dari kolom mentah dataset ERA5 ke nama variabel meteorologi standar
ERA5_FEATURES_MAPPING = {
    'temperature_era5': 'temperature',
    'humidity_era5': 'humidity',
    'dewpoint_era5': 'dewpoint',
    'rain_mm': 'rainrate',
    'sealevel_pressure_era5': 'pressure',
    'u10_era5': 'era5_u_wind',
    'v10_era5': 'era5_v_wind',
    'cloud_cover_era5': 'era5_cloud_cover',
    'cape_era5': 'era5_cape',
    'total_column_water_vapour_era5': 'era5_tcwv',
    'moisture_divergence_era5': 'era5_moisture_div',
    'direct_radiation_era5': 'era5_direct_rad',
    'sunshine_duration_era5': 'era5_sunshine'
}

# --- Tahap 3: Mesin Evaluasi Terpadu (*ModelEvaluator*) ---
# Sel 3: Mesin Evaluasi Terpadu (Unified Evaluation Engine)
def fit_best_calibrator(y_val, val_probs):
    # Platt
    platt = LogisticRegression()
    try:
        platt.fit(val_probs.reshape(-1, 1), y_val)
        platt_probs = platt.predict_proba(val_probs.reshape(-1, 1))[:, 1]
        platt_brier = brier_score_loss(y_val, platt_probs)
    except:
        platt = None
        platt_brier = 999
    # Isotonic
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(val_probs, y_val)
        iso_probs = iso.predict(val_probs)
        iso_brier = brier_score_loss(y_val, iso_probs)
    except:
        iso = None
        iso_brier = 999
    raw_brier = brier_score_loss(y_val, val_probs)
    logger.info(f"[Calibrator Selection] Brier scores - Raw: {raw_brier:.4f}, Platt: {platt_brier:.4f}, Isotonic: {iso_brier:.4f}")
    if iso_brier <= platt_brier and iso_brier < raw_brier:
        logger.info("[Calibrator Selection] Selected Isotonic Calibrator")
        return iso, "Isotonic"
    elif platt_brier < iso_brier and platt_brier < raw_brier:
        logger.info("[Calibrator Selection] Selected Platt Calibrator")
        return platt, "Platt"
    else:
        logger.info("[Calibrator Selection] Selected Raw Calibrator (No calibration applied)")
        return None, "Raw"
def apply_calibrator(calibrator, calibrator_type, probs):
    if calibrator_type == "Isotonic":
        return calibrator.predict(probs)
    elif calibrator_type == "Platt":
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    else:
        return probs.copy()
def compute_ece_mce(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(y_true)
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    for i in range(n_bins):
        mask = (y_prob_np >= bins[i]) & (y_prob_np < bins[i+1])
        if mask.sum() == 0:
            continue
        acc  = y_true_np[mask].mean()
        conf = y_prob_np[mask].mean()
        err  = abs(acc - conf)
        ece += (mask.sum() / n) * err
        mce  = max(mce, err)
    return float(ece), float(mce)
class ModelEvaluator:
    def __init__(self, base_dir=None, phase=None):
        self.base_dir = base_dir if base_dir is not None else globals().get('BASE_OUTPUT_DIR')
        self.phase = phase # 'pretraining' or 'finetuning'
        if phase:
            self.dirs = {
                'metrics': self.base_dir / self.phase,
                'plots': self.base_dir / self.phase / 'figures',
                'reports': self.base_dir / self.phase,
            }
        else:
            self.dirs = {
                'metrics': self.base_dir / 'metrics',
                'plots': self.base_dir / 'plots',
                'reports': self.base_dir / 'reports',
            }
        self.dirs['models'] = self.base_dir / 'models'
        self.dirs['models_clf'] = self.base_dir / 'models' / 'classifier'
        self.dirs['models_reg'] = self.base_dir / 'models' / 'regressor'
        self.dirs['models_cal'] = self.base_dir / 'models' / 'calibration'
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.clf_metrics = {}
        self.meteo_metrics = {}
        self.reg_metrics = {}
    def calc_meteorological_metrics(self, y_true, y_pred):
        hits = np.sum((y_pred == 1) & (y_true == 1))
        misses = np.sum((y_pred == 0) & (y_true == 1))
        false_alarms = np.sum((y_pred == 1) & (y_true == 0))
        correct_negatives = np.sum((y_pred == 0) & (y_true == 0))
        total = len(y_true)
        hits_random = ((hits + misses) * (hits + false_alarms)) / (total + 1e-9)
        csi = hits / (hits + misses + false_alarms + 1e-9)
        pod = hits / (hits + misses + 1e-9)
        far = false_alarms / (hits + false_alarms + 1e-9)
        ets = (hits - hits_random) / (hits + misses + false_alarms - hits_random + 1e-9)
        hss = 2 * (hits * correct_negatives - misses * false_alarms) / ((hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (false_alarms + correct_negatives) + 1e-9)
        return {'CSI': float(csi), 'POD': float(pod), 'FAR': float(far), 'ETS': float(ets), 'HSS': float(hss)}
    def calc_hydrological_metrics(self, y_true, y_pred):
        if len(y_true) < 2: return {'NSE': 0.0, 'KGE': 0.0, 'Correlation': 0.0}
        mean_obs, mean_sim = np.mean(y_true), np.mean(y_pred)
        std_obs, std_sim = np.std(y_true), np.std(y_pred)
        numerator = np.sum((y_true - y_pred)**2)
        denominator = np.sum((y_true - mean_obs)**2)
        nse = 1 - (numerator / (denominator + 1e-9))
        if std_obs > 0 and std_sim > 0:
            r = np.corrcoef(y_true, y_pred)[0, 1]
            alpha = std_sim / (std_obs + 1e-9)
            beta = mean_sim / (mean_obs + 1e-9)
            kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (alpha - 1)**2)
        else:
            r, kge = 0.0, 0.0
        return {'NSE': float(nse), 'KGE': float(kge), 'Correlation': float(r)}
    def evaluate_classification(self, y_true, y_prob_uncal, y_prob_cal, y_pred):
        logger.info("Menghitung metrik klasifikasi...")
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob_cal)
        brier_uncal = brier_score_loss(y_true, y_prob_uncal)
        brier_cal = brier_score_loss(y_true, y_prob_cal)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob_cal)
        pr_auc = auc(recall_arr, precision_arr)
        self.clf_metrics = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1 Score': float(f1),
            'ROC-AUC': float(roc_auc),
            'PR-AUC': float(pr_auc),
            'Brier Score (Uncalibrated)': float(brier_uncal),
            'Brier Score (Calibrated)': float(brier_cal)
        }
        self.meteo_metrics = self.calc_meteorological_metrics(y_true, y_pred)
        self.plot_calibration_curve(y_true, y_prob_uncal, y_prob_cal)
        self.plot_reliability_diagram(y_true, y_prob_uncal, y_prob_cal)
        self.plot_probability_distribution(y_prob_cal)
        self.plot_roc_curve(y_true, y_prob_cal)
        self.plot_precision_recall_curve(y_true, y_prob_cal)
        self.plot_threshold_analysis(y_true, y_prob_cal)
        self.save_combined_metrics()
        return self.clf_metrics, self.meteo_metrics
    def evaluate_regression(self, y_true, y_pred):
        logger.info("Menghitung metrik regresi...")
        if len(y_true) == 0: return {}
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        hydro = self.calc_hydrological_metrics(y_true, y_pred)
        self.reg_metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2), 'Bias': float(bias)}
        self.reg_metrics.update(hydro)
        self.plot_prediction_vs_observation(y_true, y_pred)
        self.plot_error_distribution(y_true, y_pred)
        residuals = y_true - y_pred
        self.plot_residual_distribution(residuals)
        self.plot_qq_plot(residuals)
        self.plot_rainfall_event_comparison(y_true, y_pred)
        self.save_combined_metrics()
        return self.reg_metrics
    def save_combined_metrics(self):
        combined = {
            'classification': self.clf_metrics,
            'meteorological': self.meteo_metrics,
            'regression': self.reg_metrics
        }
        with open(self.dirs['metrics'] / 'metrics.json', 'w') as f:
            json.dump(combined, f, indent=4)
    def plot_reliability_diagram(self, y_true, y_prob_uncal, y_prob_cal):
        plt.figure(figsize=(10,8))
        if y_prob_cal is not None:
            prob_true_c, prob_pred_c = calibration_curve(y_true, y_prob_cal, n_bins=10)
            plt.plot(prob_pred_c, prob_true_c, marker='s', label='Calibrated')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('Reliability Diagram')
        plt.xlabel('Prediksi Probabilitas')
        plt.ylabel('Fraksi Positif Aktual')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'reliability_diagram.png', dpi=300)
        plt.show()
        plt.close()
        plt.close()
    def plot_calibration_curve(self, y_true, y_prob_uncal, y_prob_cal=None):
        plt.figure(figsize=(10,8))
        prob_true, prob_pred = calibration_curve(y_true, y_prob_uncal, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated Model', color='blue')
        if y_prob_cal is not None:
            prob_true_c, prob_pred_c = calibration_curve(y_true, y_prob_cal, n_bins=10)
            plt.plot(prob_pred_c, prob_true_c, marker='s', label='Calibrated Model', color='green')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('Calibration Curve')
        plt.xlabel('Prediksi Probabilitas')
        plt.ylabel('Fraksi Positif Aktual')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'calibration_curve.png', dpi=300)
        plt.show()
        plt.close()
    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix.png', suffix=''):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix {suffix}')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / filename, dpi=300)
        plt.show()
        plt.close()
    def plot_roc_curve(self, y_true, y_prob):
        plt.figure(figsize=(10,8))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC={auc(fpr, tpr):.4f})')
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'roc_curve.png', dpi=300)
        plt.show()
        plt.close()
    def plot_precision_recall_curve(self, y_true, y_prob):
        plt.figure(figsize=(10,8))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AUC={auc(recall, precision):.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'precision_recall_curve.png', dpi=300)
        plt.show()
        plt.close()
    def plot_threshold_analysis(self, y_true, y_prob):
        plt.figure(figsize=(10,6))
        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        plt.plot(thresholds, f1_scores, marker='x', color='blue')
        plt.title('Threshold Analysis (F1 Score)')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'threshold_analysis.png', dpi=300)
        plt.show()
        plt.close()
    def plot_prediction_vs_observation(self, y_true, y_pred):
        plt.figure(figsize=(10,8))
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
        plt.title('Prediction vs Observation')
        plt.xlabel('Observasi (mm/jam)')
        plt.ylabel('Prediksi (mm/jam)')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'prediction_vs_observation.png', dpi=300)
        plt.show()
        plt.close()
    def plot_residual_distribution(self, residuals):
        plt.figure(figsize=(10,6))
        sns.histplot(residuals, kde=True, color='red')
        plt.title('Residual Distribution')
        plt.xlabel('Residual (Observasi - Prediksi)')
        plt.ylabel('Kepadatan')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'residual_distribution.png', dpi=300)
        plt.show()
        plt.close()
        plt.close()
        plt.close()
    def plot_timeseries_prediction_vs_observation(self, y_true, y_pred, timestamps):
        plt.figure(figsize=(15,6))
        plt.plot(timestamps, y_true, label='Observasi', color='blue', alpha=0.7)
        plt.plot(timestamps, y_pred, label='Prediksi', color='red', alpha=0.7)
        plt.title('Timeseries Prediction vs Observation')
        plt.xlabel('Waktu')
        plt.ylabel('Curah Hujan (mm/jam)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'timeseries_prediction_vs_observation.png', dpi=300)
        plt.show()
        plt.close()
    def plot_rainfall_event_comparison(self, y_true, y_pred):
        plt.figure(figsize=(12,6))
        indices = np.arange(len(y_true))
        plt.bar(indices - 0.2, y_true, width=0.4, label='Observasi', color='blue', alpha=0.7)
        plt.bar(indices + 0.2, y_pred, width=0.4, label='Prediksi', color='red', alpha=0.7)
        plt.title('Rainfall Event Intensity Comparison')
        plt.xlabel('Indeks Sampel Hujan')
        plt.ylabel('Intensitas (mm/jam)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'rainfall_event_comparison.png', dpi=300)
        plt.show()
        plt.close()
    def plot_cumulative_rainfall(self, y_true, y_pred, timestamps):
        plt.figure(figsize=(15,6))
        plt.plot(timestamps, np.cumsum(y_true), label='Cumulative Observasi', color='blue')
        plt.plot(timestamps, np.cumsum(y_pred), label='Cumulative Prediksi', color='red')
        plt.title('Cumulative Rainfall Over Time')
        plt.xlabel('Waktu')
        plt.ylabel('Akumulasi Curah Hujan (mm)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'cumulative_rainfall.png', dpi=300)
        plt.show()
        plt.close()
    def generate_report(self, n_train, n_val, n_test, feature_names, clf_metrics, meteo_metrics, reg_metrics, algorithm_name):
        thresh = globals().get('RAIN_THRESHOLD', 0.2)
        features_list = "\n".join([f"- {f}" for f in feature_names])
        report_content = f"""# Laporan Evaluasi Model Prediksi Curah Hujan Akumulasi 1-Jam

# --- Deskripsi Model ---
Framework ini membangun arsitektur dua tahap (*Two-Stage Model*) untuk memprediksi curah hujan akumulasi 1-jam.
Tahap pertama mengklasifikasikan kejadian hujan (P(Hujan) >= {thresh} mm), kemudian tahap kedua (regresi) menebak nilai intensitas untuk wilayah yang diklasifikasikan hujan.
**Algoritma Inti:** {algorithm_name}

# --- Jumlah Data ---
- **Jumlah Data Latih**: {n_train}
- **Jumlah Data Validasi**: {n_val}
- **Jumlah Data Uji**: {n_test}

# --- Daftar Variabel ---
{features_list}

# --- Hasil Klasifikasi ---
- Accuracy: {clf_metrics.get('Accuracy', 0):.4f}
- Precision: {clf_metrics.get('Precision', 0):.4f}
- Recall: {clf_metrics.get('Recall', 0):.4f}
- F1 Score: {clf_metrics.get('F1 Score', 0):.4f}
- ROC-AUC: {clf_metrics.get('ROC-AUC', 0):.4f}
- PR-AUC: {clf_metrics.get('PR-AUC', 0):.4f}
- Brier Score (Uncalibrated): {clf_metrics.get('Brier Score (Uncalibrated)', 0):.4f}
- Brier Score (Calibrated): {clf_metrics.get('Brier Score (Calibrated)', 0):.4f}

# --- Hasil Metrik Meteorologi ---
- CSI (Critical Success Index): {meteo_metrics.get('CSI', 0):.4f}
- POD (Probability of Detection): {meteo_metrics.get('POD', 0):.4f}
- FAR (False Alarm Ratio): {meteo_metrics.get('FAR', 0):.4f}
- ETS (Equitable Threat Score): {meteo_metrics.get('ETS', 0):.4f}
- HSS (Heidke Skill Score): {meteo_metrics.get('HSS', 0):.4f}

# --- Hasil Regresi ---
*(Dihitung eksklusif pada sampel aktual yang mengalami hujan)*
- RMSE (Root Mean Squared Error): {reg_metrics.get('RMSE', 0):.4f} mm
- MAE (Mean Absolute Error): {reg_metrics.get('MAE', 0):.4f} mm
- R² (Koefisien Determinasi): {reg_metrics.get('R2', 0):.4f}
- Bias: {reg_metrics.get('Bias', 0):.4f}
- Correlation: {reg_metrics.get('Correlation', 0):.4f}

# --- Analisis Feature Importance ---
*Silakan merujuk ke folder `figures/` untuk melihat visualisasi SHAP dan diagram kepentingan fitur (Feature Importance) terhadap prediksi model.*
"""
        with open(self.dirs['reports'] / 'classification_report.txt', 'w') as f:
            f.write(report_content)

# --- FASE 1: PRE-TRAINING (Data Satelit & ERA5) ---
# Sel 4: Pemuatan Data dan Penyatuan (Data Loading & Merge)
# Menggabungkan data historis cuaca ERA5 dengan target curah hujan satelit GSMaP untuk pre-training.
# Sel 4: Pemuatan Data dan Penyatuan (Data Loading & Merge)
logger.info("Loading ERA5 + GSMaP dataset...")

# PATH

if os.path.exists('/kaggle/input'):
    search_file = glob.glob('/kaggle/input/**/ERA5_Hourly_All_Requested_Features_2000_2026.csv', recursive=True)
    SAT_DIR = Path(search_file[0]).parent if search_file else Path('/kaggle/input/datasets/jerismeteo/google-earth-engine-data')
else:
    BASE_DATA_DIR = Path('D:/Github/Projek_Rainfall')
    SAT_DIR = BASE_DATA_DIR / 'Google_Earth_Engine' / 'Data_Satelit'

# LOAD ERA5

df_era5 = pd.read_csv(SAT_DIR / "ERA5_Hourly_All_Requested_Features_2000_2026.csv")
if "datetime_utc" in df_era5.columns:
    df_era5["timestamp"] = pd.to_datetime(df_era5["datetime_utc"], utc=True)
else:
    df_era5["timestamp"] = pd.to_datetime(df_era5["unixtime"], unit="s", utc=True)
df_era5 = df_era5.set_index("timestamp").sort_index()

# LOAD GSMAP (Hourly)

df_gsmap = pd.read_csv(SAT_DIR / "Rainfall_GSMap_TimeSeries_UNIX.csv")
if "datetime_utc" in df_gsmap.columns:
    df_gsmap["timestamp"] = pd.to_datetime(df_gsmap["datetime_utc"], utc=True)
else:
    df_gsmap["timestamp"] = pd.to_datetime(df_gsmap["unixtime"], unit="s", utc=True)
df_gsmap = df_gsmap.set_index("timestamp").sort_index()

# RESAMPLE GSMAP & ALIGN TO HOURLY (Rename column to mimic precipitation)

logger.info("Resampling GSMaP to hourly rainfall...")
df_gsmap_hourly = (
    df_gsmap[["hourlyPrecipRateGC"]]
    .rename(columns={"hourlyPrecipRateGC": "precipitation"})
)

# MERGE

df_merged = df_era5.merge(
    df_gsmap_hourly,
    left_index=True,
    right_index=True,
    how="left"
)

# FEATURE MAPPING

for raw_col, new_name in ERA5_FEATURES_MAPPING.items():
    if raw_col in df_merged.columns:
        df_merged[new_name] = df_merged[raw_col]

# REPLACE ERA5 RAIN WITH IMERG

if "rainrate" in df_merged.columns and "precipitation" in df_merged.columns:
    logger.info("Replacing ERA5 rainfall using GSMaP...")
    df_merged["rainrate"] = df_merged["precipitation"].combine_first(df_merged["rainrate"])

# REMOVE DRIZZLE

if "rainrate" in df_merged.columns:
    df_merged["rainrate"] = np.where(df_merged["rainrate"] >= 0.2, df_merged["rainrate"], 0.0)

# KEEP REQUIRED FEATURES

selected_features = [v for v in ERA5_FEATURES_MAPPING.values() if v in df_merged.columns]
df_merged = df_merged[selected_features]

# CLEAN DATA

if "temperature" in df_merged.columns:
    df_hourly = df_merged.dropna(subset=["temperature"]).copy()
else:
    df_hourly = df_merged.dropna().copy()
df_hourly = df_hourly.select_dtypes(include=[np.number])
# Visualisasi Distribusi Hujan vs Tidak Hujan (PRE-TRAINING)
# Memplot persentase kejadian hujan (rainrate >= 0.2 mm/jam) pada data pre-training.
rain_count = (df_hourly['rainrate'] > 0).sum()
no_rain_count = (df_hourly['rainrate'] == 0).sum()
plt.figure(figsize=(6, 4))
bars = plt.bar(['Tidak Hujan', 'Hujan'], [no_rain_count, rain_count], color=['#1f77b4', '#ff7f0e'])

# --- Tahap 4: Rekayasa Fitur (*Feature Engineering*) ---
# Sel 5: Rekayasa Fitur Fisika Atmosfer
# Menghitung indeks diurnal harian dan fitur atmosfer penting seperti dewpoint depression.
def engineer_features(df_base):
    logger.info("Mengekstraksi fitur fisika & temporal...")
    cols_diff = ['temperature', 'dewpoint', 'pressure']
    for c in cols_diff:
        if c in df_base.columns:
            df_base[f'{c}_tendency_1h'] = df_base[c].diff()
    if 'temperature' in df_base.columns and 'dewpoint' in df_base.columns:
        df_base['dewpoint_depression'] = df_base['temperature'] - df_base['dewpoint']
    # Fitur waktu siklikal (cyclic time encoding)
    df_base['sin_hour']  = np.sin(2 * np.pi * df_base.index.hour / 24.0)
    df_base['cos_hour']  = np.cos(2 * np.pi * df_base.index.hour / 24.0)
    df_base['sin_month'] = np.sin(2 * np.pi * df_base.index.month / 12.0)
    df_base['cos_month'] = np.cos(2 * np.pi * df_base.index.month / 12.0)
    df_base['sin_doy']   = np.sin(2 * np.pi * df_base.index.dayofyear / 365.25)
    df_base['cos_doy']   = np.cos(2 * np.pi * df_base.index.dayofyear / 365.25)
    df = df_base.copy()
    df = df.dropna()
    logger.info(f"Dimensi akhir setelah rekayasa fitur: {df.shape}")
    return df

# --- Tahap 4.5: Persiapan Parameter LSTM ---
# Sel 6: Setup Parameter Dasar & Split Data LSTM
# Memformat data pre-training menjadi array sekuensial 3D (samples, timesteps, features)
# menggunakan lookback window LOOKBACK=24 jam untuk melatih RNN/LSTM.
logger.info(f"\n{'='*50}\nMEMULAI TRAINING LSTM 1-JAM (OPTUNA HPO)\n{'='*50}")
evaluator = ModelEvaluator()
df_features = engineer_features(df_hourly)
# Target: curah hujan 1 jam ke depan
df_features['target_amount'] = df_features['rainrate'].shift(-1)
df_features = df_features.dropna()
df_features['target_occurrence'] = (df_features['target_amount'] >= RAIN_THRESHOLD).astype(int)
# 3. Data Split
if RUN_MODE == "PIPELINE_TEST":
    n = len(df_features)
    train_mask = np.arange(n) < int(n * 0.7)
    val_mask   = (np.arange(n) >= int(n * 0.7)) & (np.arange(n) < int(n * 0.9))
    test_mask  = np.arange(n) >= int(n * 0.9)
else:
    # [PEMBAGIAN DATA PRE-TRAINING BERDASARKAN TANGGAL]
    train_mask = (df_features.index >= PRETRAIN__TRAIN_START) & (df_features.index <= PRETRAIN__TRAIN_END)
    val_mask   = (df_features.index >= PRETRAIN__VAL_START)   & (df_features.index < PRETRAIN__VAL_END)
    test_mask  = (df_features.index >= PRETRAIN__TEST_START)  & (df_features.index <= PRETRAIN__TEST_END)
X     = df_features.drop(columns=['target_amount', 'target_occurrence'])
y_occ = df_features['target_occurrence']
y_reg = df_features['target_amount']
scaler = MinMaxScaler()
scaler.fit(X[train_mask])
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
joblib.dump(scaler, evaluator.dirs['models'] / 'pretrain_scaler.pkl')
logger.info(f"Scaler di-fit HANYA pada {X[train_mask].shape[0]} sampel latih. Disimpan ke pretrain_scaler.pkl")
def make_3d(X_df, y_o, y_r, lookback):
    X_arr, yo_arr, yr_arr = [], [], []
    for i in range(len(X_df) - lookback + 1):
        X_arr.append(X_df.iloc[i:i+lookback].values)
        yo_arr.append(y_o.iloc[i+lookback-1])
        yr_arr.append(y_r.iloc[i+lookback-1])
    return (np.array(X_arr, dtype=np.float32),
            np.array(yo_arr, dtype=np.float32),
            np.array(yr_arr, dtype=np.float32))
X_tr_seq, y_tr_o, y_tr_r = make_3d(X_scaled[train_mask], y_occ[train_mask], y_reg[train_mask], LOOKBACK)
X_va_seq, y_va_o, y_va_r = make_3d(X_scaled[val_mask],   y_occ[val_mask],   y_reg[val_mask],   LOOKBACK)
X_te_seq, y_te_o, y_te_r = make_3d(X_scaled[test_mask],  y_occ[test_mask],  y_reg[test_mask],  LOOKBACK)
logger.info(f"Sequence shapes — Train: {X_tr_seq.shape}, Val: {X_va_seq.shape}, Test: {X_te_seq.shape}")
logger.info(f"Distribusi Hujan (Train): {y_tr_o.sum():.0f}/{len(y_tr_o)} ({100*y_tr_o.mean():.1f}%)")
logger.info(f"Distribusi Hujan (Val)  : {y_va_o.sum():.0f}/{len(y_va_o)} ({100*y_va_o.mean():.1f}%)")
logger.info(f"Distribusi Hujan (Test) : {y_te_o.sum():.0f}/{len(y_te_o)} ({100*y_te_o.mean():.1f}%)")
cek_proporsi_pretraining(X_tr_seq, y_tr_o, X_va_seq, y_va_o, X_te_seq, y_te_o)
# Simpan salinan split pre-trained sebelum ditimpa di fase fine-tuning
X_te_seq_pre = X_te_seq.copy()
y_te_o_pre = y_te_o.copy()
y_te_r_pre = y_te_r.copy()

# --- Tahap 5: Pelatihan Klasifikasi Hujan (Optuna LSTM) ---
# Sel 7: Optuna Tuning untuk Classifier LSTM
# Mengoptimalkan hyperparameter arsitektur BiLSTM Classifier menggunakan Optuna.
def build_lstm_clf(lookback, units_l1, units_l2, lr, dropout):
    model = Sequential([
        Input(shape=(lookback, X.shape[1])),
        LSTM(units_l1, return_sequences=True),
        Dropout(dropout),
        LSTM(units_l2, return_sequences=False),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model
def objective_clf(trial):
    K.clear_session()
    gc.collect()
    units_l1   = trial.suggest_categorical('units_l1', [64, 128])
    units_l2   = trial.suggest_categorical('units_l2', [32, 64])
    lr         = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    dropout    = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    model = build_lstm_clf(LOOKBACK, units_l1, units_l2, lr, dropout)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_tr_seq, y_tr_o, validation_data=(X_va_seq, y_va_o),
              epochs=EPOCHS_HPO, batch_size=batch_size, callbacks=[es], verbose=0)
    val_prob = model.predict(X_va_seq, verbose=0).flatten()
    # Dynamic threshold search — pilih threshold dengan F1 terbaik di validasi
    best_f1_trial, best_thresh_trial = 0.0, 0.5
    for _t in np.arange(0.20, 0.70, 0.02):
        _preds = (val_prob >= _t).astype(int)
        _f1 = f1_score(y_va_o, _preds, zero_division=0)
        if _f1 > best_f1_trial:
            best_f1_trial, best_thresh_trial = _f1, _t
    trial.set_user_attr('best_threshold', float(best_thresh_trial))
    return best_f1_trial
logger.info("Memulai HPO Optuna untuk LSTM Classifier")
optuna.logging.set_verbosity(optuna.logging.INFO)
# [AUDIT FIX] MedianPruner menghentikan trial tidak menjanjikan lebih awal
db_path = evaluator.dirs['models'] / 'optuna_lstm.db'
os.makedirs(db_path.parent, exist_ok=True)
storage_url = f"sqlite:///{db_path.as_posix()}"
study_clf = optuna.create_study(
    study_name="lstm_classifier",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    storage=storage_url,
    direction='maximize',
    load_if_exists=True
)
study_clf.optimize(objective_clf, n_trials=OPTUNA_TRIALS)
best_clf_params = study_clf.best_params
logger.info(f"Parameter terbaik Classifier: {best_clf_params}")
K.clear_session()
gc.collect()
clf_best = build_lstm_clf(LOOKBACK,
                          best_clf_params['units_l1'],
                          best_clf_params['units_l2'],
                          best_clf_params['lr'],
                          best_clf_params['dropout'])
# ModelCheckpoint menyimpan bobot terbaik selama pelatihan akhir (post-HPO)
clf_ckpt_path = str(evaluator.dirs['models_clf'] / 'best_lstm_occ_ckpt.keras')
es        = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
ckpt      = ModelCheckpoint(clf_ckpt_path, monitor='val_loss', save_best_only=True, verbose=0)
history_clf = clf_best.fit(
    X_tr_seq, y_tr_o,
    validation_data=(X_va_seq, y_va_o),
    epochs=EPOCHS_BEST,
    batch_size=best_clf_params['batch_size'],
    callbacks=[es, reduce_lr, ckpt],
    verbose=1
)
plt.figure(figsize=(10,6))
plt.plot(history_clf.history['loss'], label='Train')
plt.plot(history_clf.history['val_loss'], label='Validation', color='orange')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(evaluator.dirs['plots'] / 'training_loss_clf.png', dpi=300)
plt.show()
clf_best.save(evaluator.dirs['models_clf'] / 'best_lstm_occ.keras')
# Plot arsitektur model (Graphviz)
try:
    _arch_path_clf = str(evaluator.dirs['models_clf'] / 'model_architecture_clf.png')
    plot_model(clf_best, to_file=_arch_path_clf, show_shapes=True, show_layer_names=True, dpi=96)
    logger.info(f"Arsitektur Classifier disimpan: {_arch_path_clf}")
    display(Image(_arch_path_clf))
except Exception as _e:
    logger.warning(f"plot_model tidak tersedia (Graphviz mungkin belum terinstall): {_e}")
# Kalibrasi Probabilitas — Isotonic vs Platt Scaling
# Kalibrasi WAJIB menggunakan data Validasi
val_probs_uncal = clf_best.predict(X_va_seq, verbose=0).flatten()
best_calibrator_pretrain, cal_method_pretrain = fit_best_calibrator(y_va_o, val_probs_uncal)
if best_calibrator_pretrain is not None:
    joblib.dump(best_calibrator_pretrain, evaluator.dirs['models_cal'] / 'isotonic_calibrator.pkl')
# Hitung nilai Brier untuk pencatatan metadata
iso_temp = IsotonicRegression(out_of_bounds='clip').fit(val_probs_uncal, y_va_o)
brier_iso = brier_score_loss(y_va_o, iso_temp.predict(val_probs_uncal))
platt_temp = LogisticRegression().fit(val_probs_uncal.reshape(-1, 1), y_va_o)
brier_platt = brier_score_loss(y_va_o, platt_temp.predict_proba(val_probs_uncal.reshape(-1, 1))[:, 1])
joblib.dump({'method': cal_method_pretrain, 'brier_iso': brier_iso, 'brier_platt': brier_platt},
            evaluator.dirs['models_cal'] / 'calibration_meta.pkl')
# Ambil threshold optimal dari trial terbaik Optuna
best_threshold_pretrain = study_clf.best_trial.user_attrs.get('best_threshold', 0.5)
joblib.dump(best_threshold_pretrain, evaluator.dirs['models_clf'] / 'best_threshold_pretrain.pkl')
logger.info(f"Optimal Classification Threshold (Pre-training): {best_threshold_pretrain:.2f}")
logger.info("Pre-training Classifier selesai. ECE/MCE tersedia secara global.")

# --- Tahap 6: Pelatihan Regresi Intensitas (Optuna LSTM) ---
# Sel 8: Optuna Tuning untuk Regressor LSTM
# Mengoptimalkan hyperparameter arsitektur BiLSTM Regressor pada data dengan intensitas positif.
# Hanya melatih data yang aktualnya hujan
rainy_tr = y_tr_o == 1
rainy_va = y_va_o == 1
if np.sum(rainy_tr) > 10:
    X_tr_r_val, y_tr_r_val = X_tr_seq[rainy_tr], np.log1p(y_tr_r[rainy_tr])
    X_va_r_val, y_va_r_val = X_va_seq[rainy_va], np.log1p(y_va_r[rainy_va])
    def build_lstm_reg(lookback, units_l1, units_l2, lr, dropout):
        model = Sequential([
            Input(shape=(lookback, X.shape[1])),
            LSTM(units_l1, return_sequences=True),
            Dropout(dropout),
            LSTM(units_l2, return_sequences=False),
            Dropout(dropout),
            Dense(1, activation='relu')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model
    def objective_reg(trial):
        K.clear_session()
        gc.collect()
        units_l1   = trial.suggest_categorical('units_l1', [64, 128])
        units_l2   = trial.suggest_categorical('units_l2', [32, 64])
        lr         = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        dropout    = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
        batch_size = trial.suggest_categorical('batch_size', [64, 128,256])
        model = build_lstm_reg(LOOKBACK, units_l1, units_l2, lr, dropout)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_tr_r_val, y_tr_r_val, validation_data=(X_va_r_val, y_va_r_val),
                  epochs=EPOCHS_HPO, batch_size=batch_size, callbacks=[es], verbose=0)
        val_loss = model.evaluate(X_va_r_val, y_va_r_val, verbose=0)
        return val_loss
    logger.info("Memulai HPO Optuna untuk LSTM Regressor (BiLSTM)...")
    # MedianPruner untuk efisiensi HPO regressor
    db_path = evaluator.dirs['models'] / 'optuna_lstm.db'
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study_reg = optuna.create_study(
        study_name="lstm_regressor",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=storage_url,
        direction='minimize',
        load_if_exists=True
    )
    study_reg.optimize(objective_reg, n_trials=OPTUNA_TRIALS)
    best_reg_params = study_reg.best_params
    logger.info(f"Parameter terbaik Regressor: {best_reg_params}")
    K.clear_session()
    gc.collect()
    reg_best = build_lstm_reg(LOOKBACK,
                              best_reg_params['units_l1'],
                              best_reg_params['units_l2'],
                              best_reg_params['lr'],
                              best_reg_params['dropout'])
    # ModelCheckpoint untuk pelatihan akhir regressor
    reg_ckpt_path = str(evaluator.dirs['models_reg'] / 'best_lstm_reg_ckpt.keras')
    es        = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ckpt_reg  = ModelCheckpoint(reg_ckpt_path, monitor='val_loss', save_best_only=True, verbose=0)
    history_reg = reg_best.fit(
        X_tr_r_val, y_tr_r_val,
        validation_data=(X_va_r_val, y_va_r_val),
        epochs=EPOCHS_BEST,
        batch_size=best_reg_params['batch_size'],
        callbacks=[es, reduce_lr, ckpt_reg],
        verbose=1
    )
    plt.figure(figsize=(10,6))
    plt.plot(history_reg.history['loss'], label='Train MSE')
    plt.plot(history_reg.history['val_loss'], label='Val MSE', color='red')
    plt.title('Training and Validation Loss (Regressor)')
    plt.legend()
    plt.savefig(evaluator.dirs['plots'] / 'training_loss_reg.png', dpi=300)
    plt.show()
    reg_best.save(evaluator.dirs['models_reg'] / 'best_lstm_reg.keras')
    # Plot arsitektur model (Graphviz)
    try:
        _arch_path_reg = str(evaluator.dirs['models_reg'] / 'model_architecture_reg.png')
        plot_model(reg_best, to_file=_arch_path_reg, show_shapes=True, show_layer_names=True, dpi=96)
        logger.info(f"Arsitektur Regressor disimpan: {_arch_path_reg}")
        display(Image(_arch_path_reg))
    except Exception as _e:
        logger.warning(f"plot_model tidak tersedia (Graphviz mungkin belum terinstall): {_e}")
else:
    logger.warning("Tidak cukup sampel hujan untuk melatih regressor.")
    reg_best = None

# --- FASE 2: FINE-TUNING (Data Aktual Stasiun AWS) ---
# **Tujuan:** Menyelaraskan model dengan kondisi iklim lokal.
# Sel ini menukar sumber data dasar menjadi **data pengukuran asli dari Stasiun AWS**, sembari mempertahankan fitur Angin dan Konvektif yang dipinjam secara dinamis dari ERA5.
# Sel 8.5: Pemuatan Data Aktual Stasiun Bumi AWS (Fine-Tuning)
# Memuat data stasiun AWS bumi dan memperbarui fitur dasar ERA5 menggunakan data observasi darat.
# Pemuatan Data dan Penyatuan (Data Loading & Merge)
logger.info("Mendeteksi Environment & Memuat data AWS untuk Fine-Tuning...")
if os.path.exists('/kaggle/input'):
    sat_search = glob.glob('/kaggle/input/**/ERA5_Hourly_All_Requested_Features_2000_2026.csv', recursive=True)
    station_search = glob.glob('/kaggle/input/**/id-05_clear_data_hourly.csv', recursive=True)
    if sat_search:
        SAT_DIR = Path(sat_search[0]).parent
    else:
        SAT_DIR = Path('/kaggle/input/google-earth-engine-data/Google_Earth_Engine/Data_Satelit')
    if station_search:
        STATION_FILE = Path(station_search[0])
    else:
        STATION_FILE = Path('/kaggle/input/google-earth-engine-data/Google_Earth_Engine/Data_Stasiun/id-05_clear_data_hourly.csv')
else:
    BASE_DATA_DIR = Path('D:/Github/Projek_Rainfall')
    STATION_FILE  = BASE_DATA_DIR / 'Google_Earth_Engine' / 'Data_Satelit' / 'id-05_clear_data_hourly.csv'
    SAT_DIR       = BASE_DATA_DIR / 'Google_Earth_Engine' / 'Data_Satelit'
# 1. Memuat Data Stasiun AWS Lokal
df_station = pd.read_csv(STATION_FILE)
df_station['timestamp'] = pd.to_datetime(df_station['datetime_utc'], utc=True)
df_station = df_station.set_index('timestamp').sort_index()
cols_to_keep = ['temperature', 'humidity', 'pressure', 'dewpoint', 'rainrate']
df_station = df_station[[c for c in cols_to_keep if c in df_station.columns]]
# 2. Memuat ERA5
df_era5 = pd.read_csv(SAT_DIR / 'ERA5_Hourly_All_Requested_Features_2000_2026.csv')
if 'datetime_utc' in df_era5.columns:
    df_era5['timestamp'] = pd.to_datetime(df_era5['datetime_utc'], utc=True)
else:
    df_era5['timestamp'] = pd.to_datetime(df_era5['unixtime'], unit='s', utc=True)
df_era5 = df_era5.set_index('timestamp').sort_index()
df_merged = df_era5.copy()
for raw_col, new_name in ERA5_FEATURES_MAPPING.items():
    if raw_col in df_era5.columns:
        df_merged[new_name] = df_era5[raw_col]
# Hapus noise drizzle efek ERA5 (WMO wet-day threshold 0.1 mm/h)
if 'rainrate' in df_merged.columns:
    df_merged['rainrate'] = np.where(df_merged['rainrate'] >= 0.1, df_merged['rainrate'], 0.0)
selected_features = list(ERA5_FEATURES_MAPPING.values())
df_merged = df_merged[[c for c in selected_features if c in df_merged.columns]]
# [REPLACEMENT AWS] Di mana AWS ada, timpa ERA5
df_merged.update(df_station)
# [AUDIT FIX] Filter mulai 2023 (bukan 2025) untuk mencakup split fine-tuning 2023-2024
df_merged = df_merged[df_merged.index >= '2024-12-01']
drop_subset = ['temperature'] if 'temperature' in df_merged.columns else None
df_hourly = df_merged.dropna(subset=drop_subset).copy() if drop_subset else df_merged.dropna().copy()
df_hourly = df_hourly.select_dtypes(include=[np.number])
logger.info(f"Dimensi data terpadu (FINE-TUNING): {df_hourly.shape}")
# Visualisasi Distribusi Hujan vs Tidak Hujan (FINE-TUNING)
# Memantau proporsi kejadian hujan stasiun bumi AWS pada data fine-tuning.
rain_count = (df_hourly['rainrate'] > 0).sum()
no_rain_count = (df_hourly['rainrate'] == 0).sum()
plt.figure(figsize=(6, 4))
bars = plt.bar(['Tidak Hujan', 'Hujan'], [no_rain_count, rain_count], color=['#1f77b4', '#ff7f0e'])
plt.title('Distribusi Data Hujan vs Tidak Hujan (FINE-TUNING)', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Sampel')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.005), f'{yval:,}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Tahap 4: Rekayasa Fitur (*Feature Engineering*) ---
# Sel ini meramu data mentah 1-jaman menjadi format 1-jaman yang siap ditelan model. Fitur-fitur canggih seperti **Tren 1 Jam** (*difference*) dihitung di sini, beserta nilai agregasi fitur konvektif tingkat lanjut (CAPE, K-Index) dari ERA5.
# Sel 9: Rekayasa Fitur Fisika Atmosfer (Fine-Tuning)
# Menghitung ulang fitur temporal lag dan dewpoint depression pada data AWS.
def engineer_features(df_base):
    logger.info("Mengekstraksi fitur fisika & temporal...")
    cols_diff = ['temperature', 'dewpoint', 'pressure']
    for c in cols_diff:
        if c in df_base.columns:
            df_base[f'{c}_tendency_1h'] = df_base[c].diff()
    if 'temperature' in df_base.columns and 'dewpoint' in df_base.columns:
        df_base['dewpoint_depression'] = df_base['temperature'] - df_base['dewpoint']
    # Fitur waktu siklikal (cyclic time encoding)
    df_base['sin_hour']  = np.sin(2 * np.pi * df_base.index.hour / 24.0)
    df_base['cos_hour']  = np.cos(2 * np.pi * df_base.index.hour / 24.0)
    df_base['sin_month'] = np.sin(2 * np.pi * df_base.index.month / 12.0)
    df_base['cos_month'] = np.cos(2 * np.pi * df_base.index.month / 12.0)
    df_base['sin_doy']   = np.sin(2 * np.pi * df_base.index.dayofyear / 365.25)
    df_base['cos_doy']   = np.cos(2 * np.pi * df_base.index.dayofyear / 365.25)
    df = df_base.copy()
    df = df.dropna()
    logger.info(f"Dimensi akhir setelah rekayasa fitur: {df.shape}")
    return df

# --- Tahap 4.75: Pembentukan Sekuens Data Waktu ---
# Mengubah data tabular (2D) menjadi format sekuens (3D) `[samples, timesteps, features]` yang disyaratkan oleh arsitektur LSTM.
# Sel 9.5: Preparasi Sequence dan Split Data (FINE-TUNING)
# Memformat dataset fine-tuning menjadi sekuens temporal 3D (samples, LOOKBACK=24, features) untuk update model.
df_features = engineer_features(df_hourly)
df_features['target_amount'] = df_features['rainrate'].shift(-1)
df_features = df_features.dropna()
# [PEMBAGIAN DATA FINE-TUNING BERDASARKAN TANGGAL]
if RUN_MODE == "PIPELINE_TEST":
    n = len(df_features)
    train_mask = np.arange(n) < int(n * 0.6)
    val_mask   = (np.arange(n) >= int(n * 0.6)) & (np.arange(n) < int(n * 0.8))
    test_mask  = np.arange(n) >= int(n * 0.8)
else:
    train_mask = (df_features.index >= FINETUNE_TRAIN_START) & (df_features.index <= FINETUNE_TRAIN_END)
    val_mask   = (df_features.index >= FINETUNE_VAL_START)   & (df_features.index < FINETUNE_VAL_END)
    test_mask  = (df_features.index >= FINETUNE_TEST_START)  & (df_features.index <= FINETUNE_TEST_END)
y_all_r = df_features['target_amount'].values
y_all_o = (y_all_r >= RAIN_THRESHOLD).astype(int)
# Load pre-trained MinMaxScaler (hanya transform, tidak re-fit)
scaler = joblib.load(evaluator.dirs['models'] / 'pretrain_scaler.pkl')
X_all_df = df_features.drop(columns=['target_amount'])
# Selaraskan kolom dengan fitur pre-training; isi kolom yang hilang dengan 0
X_all_df = X_all_df.reindex(columns=scaler.feature_names_in_, fill_value=0.0)
X_scaled = scaler.transform(X_all_df)
# [AUDIT FIX] Perbaikan Alignment Mask Sekuens:
# Menggunakan pendekatan blok independen per split (identik dengan pre-training make_3d).
# Ini menggantikan create_sequences(full_array) + mask[LOOKBACK:] yang rentan offset.
X_all_pd = pd.DataFrame(X_scaled, index=df_features.index, columns=scaler.feature_names_in_)
y_o_pd   = pd.Series(y_all_o, index=df_features.index)
y_r_pd   = pd.Series(y_all_r, index=df_features.index)
def make_3d_np(X_df, y_o, y_r, lookback):
    Xv, yov, yrv = X_df.values, y_o.values, y_r.values
    X_arr, yo_arr, yr_arr = [], [], []
    for i in range(len(Xv) - lookback + 1):
        X_arr.append(Xv[i:i+lookback])
        yo_arr.append(yov[i+lookback-1])
        yr_arr.append(yrv[i+lookback-1])
    return (np.array(X_arr, dtype=np.float32),
            np.array(yo_arr, dtype=np.float32),
            np.array(yr_arr, dtype=np.float32))
X_tr_seq, y_tr_o, y_tr_r = make_3d_np(X_all_pd[train_mask], y_o_pd[train_mask], y_r_pd[train_mask], LOOKBACK)
X_v_seq,  y_v_o,  y_v_r  = make_3d_np(X_all_pd[val_mask],   y_o_pd[val_mask],   y_r_pd[val_mask],   LOOKBACK)
X_te_seq, y_te_o, y_te_r = make_3d_np(X_all_pd[test_mask],  y_o_pd[test_mask],  y_r_pd[test_mask],  LOOKBACK)
logger.info(f"Fine-Tuning Sequence — Train: {X_tr_seq.shape}, Val: {X_v_seq.shape}, Test: {X_te_seq.shape}")
logger.info(f"Distribusi Hujan (Train): {y_tr_o.sum():.0f}/{len(y_tr_o)} ({100*y_tr_o.mean():.1f}%)")
logger.info(f"Distribusi Hujan (Val)  : {y_v_o.sum():.0f}/{len(y_v_o)} ({100*y_v_o.mean():.1f}%)")
logger.info(f"Distribusi Hujan (Test) : {y_te_o.sum():.0f}/{len(y_te_o)} ({100*y_te_o.mean():.1f}%)")
cek_proporsi_finetuning(X_tr_seq, y_tr_o, X_v_seq, y_v_o, X_te_seq, y_te_o)

# --- Tahap Terakhir: Proses Fine-Tuning & Perbandingan Model (LSTM) ---
# Di tahap ini, bobot dari model LSTM global digunakan sebagai basis (Pre-trained) dan di-*retrain* secara hati-hati menggunakan dataset lokal (AWS). Laporan komparasi performa akan dicetak secara otomatis.
# Sel 13: Proses Fine-Tuning (LSTM)
# Melatih ulang model dasar (Classifier & Regressor LSTM) secara warm-start menggunakan data stasiun AWS,
# menyetel ulang kalibrator optimal pada data validasi fine-tuning.
# 1. LOAD MODEL TERBAIK PRE-TRAINING (diperlukan untuk fine-tuning)
logger.info("Memuat model terbaik pre-training...")
clf_best = tf.keras.models.load_model(evaluator.dirs['models_clf'] / 'best_lstm_occ.keras')
reg_best = None
if (evaluator.dirs['models_reg'] / 'best_lstm_reg.keras').exists():
    reg_best = tf.keras.models.load_model(evaluator.dirs['models_reg'] / 'best_lstm_reg.keras')
# 2. PROSES FINE-TUNING CLASSIFIER (Learning Rate = 0.01)
logger.info("Memulai Fine-Tuning LSTM Classifier...")
clf_best.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
batch_size_ft = 64
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (tf.data.Dataset
    .from_tensor_slices((X_tr_seq, y_tr_o))
    .shuffle(buffer_size=len(X_tr_seq), seed=SEED)
    .batch(batch_size_ft)
    .cache()
    .prefetch(AUTOTUNE))
val_ds = (tf.data.Dataset
    .from_tensor_slices((X_v_seq, y_v_o))
    .batch(batch_size_ft)
    .cache()
    .prefetch(AUTOTUNE))
clf_ft_ckpt = str(evaluator.dirs['models_clf'] / 'best_lstm_occ_finetuned_ckpt.keras')
es      = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
ckpt_ft = ModelCheckpoint(clf_ft_ckpt, monitor='val_loss', save_best_only=True, verbose=0)
history_clf_ft = clf_best.fit(train_ds, validation_data=val_ds,
                               epochs=EPOCHS_FINETUNE, callbacks=[es, ckpt_ft], verbose=1)
clf_best.save(evaluator.dirs['models_clf'] / 'best_lstm_occ_finetuned.keras')
# Kalibrasi ulang menggunakan data validasi fine-tuning
val_probs_uncal = clf_best.predict(X_v_seq, verbose=0).flatten()
iso_reg, cal_method_ft = fit_best_calibrator(y_v_o, val_probs_uncal)
if iso_reg is not None:
    joblib.dump(iso_reg, evaluator.dirs['models_cal'] / 'isotonic_calibrator_finetuned.pkl')
# Dynamic threshold search pada data validasi fine-tuning
val_probs_cal_ft = apply_calibrator(iso_reg, cal_method_ft, val_probs_uncal) if iso_reg is not None else val_probs_uncal
best_threshold_finetuned, best_f1_ft = 0.5, 0.0
for _t in np.arange(0.20, 0.70, 0.02):
    _preds = (val_probs_cal_ft >= _t).astype(int)
    _f1 = f1_score(y_v_o, _preds, zero_division=0)
    if _f1 > best_f1_ft:
        best_f1_ft, best_threshold_finetuned = _f1, float(_t)
joblib.dump(best_threshold_finetuned, evaluator.dirs['models_clf'] / 'best_threshold_finetuned.pkl')
logger.info(f"Optimal Classification Threshold (Fine-Tuning): {best_threshold_finetuned:.2f} (F1={best_f1_ft:.4f})")
rainy_tr = y_tr_o == 1
rainy_v  = y_v_o  == 1
if np.sum(rainy_tr) > 10 and reg_best is not None:
    logger.info("Memulai Fine-Tuning Regressor...")
    X_tr_r_val = X_tr_seq[rainy_tr]
    y_tr_r_val = np.log1p(y_tr_r[rainy_tr])
    X_v_r_val  = X_v_seq[rainy_v]
    y_v_r_val  = np.log1p(y_v_r[rainy_v])
    reg_best.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    train_reg_ds = (tf.data.Dataset
        .from_tensor_slices((X_tr_r_val, y_tr_r_val))
        .shuffle(buffer_size=len(X_tr_r_val), seed=SEED)
        .batch(batch_size_ft)
        .cache()
        .prefetch(AUTOTUNE))
    val_reg_ds = (tf.data.Dataset
        .from_tensor_slices((X_v_r_val, y_v_r_val))
        .batch(batch_size_ft)
        .cache()
        .prefetch(AUTOTUNE))
    reg_ft_ckpt = str(evaluator.dirs['models_reg'] / 'best_lstm_reg_finetuned_ckpt.keras')
    es_reg      = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ckpt_reg_ft = ModelCheckpoint(reg_ft_ckpt, monitor='val_loss', save_best_only=True, verbose=0)
    history_reg_ft = reg_best.fit(train_reg_ds, validation_data=val_reg_ds,
                                   epochs=EPOCHS_FINETUNE, callbacks=[es_reg, ckpt_reg_ft], verbose=1)
    reg_best.save(evaluator.dirs['models_reg'] / 'best_lstm_reg_finetuned.keras')
logger.info("Fine-Tuning LSTM Selesai!")

# --- Bab 1: Evaluasi Model Pretraining ---
# Dalam bab ini, model dasar LSTM yang dilatih hanya menggunakan data satelit (Pre-Trained) dievaluasi kinerjanya pada data uji sekuensial stasiun AWS bumi untuk melihat tingkat akurasi awal sebelum penyesuaian lokal.
# Bab 1: Evaluasi Model Pretraining (LSTM)
logger.info("Memulai Bab 1: Evaluasi Model Pretraining...")
evaluator_pre = ModelEvaluator(BASE_OUTPUT_DIR, phase='pretraining')
# Load model pre-trained asli (satelit)
clf_pretrain = tf.keras.models.load_model(evaluator_pre.dirs['models_clf'] / 'best_lstm_occ.keras')
reg_pretrain = tf.keras.models.load_model(evaluator_pre.dirs['models_reg'] / 'best_lstm_reg.keras') if (evaluator_pre.dirs['models_reg'] / 'best_lstm_reg.keras').exists() else None
# Pemuatan calibrator pre-train
calibrator_pre = joblib.load(evaluator_pre.dirs['models_cal'] / 'isotonic_calibrator.pkl')
cal_meta_pre = joblib.load(evaluator_pre.dirs['models_cal'] / 'calibration_meta.pkl')
cal_method_pre = cal_meta_pre['method']
# Load threshold optimal pre-training
_thresh_file_pre = evaluator_pre.dirs['models_clf'] / 'best_threshold_pretrain.pkl'
best_threshold_pretrain = joblib.load(_thresh_file_pre) if _thresh_file_pre.exists() else 0.5
logger.info(f"Threshold klasifikasi pre-training: {best_threshold_pretrain:.2f}")
# Prediksi Klasifikasi pada data uji satelit pre-training
pre_test_prob_uncal = clf_pretrain.predict(X_te_seq_pre, verbose=0).flatten()
pre_test_prob_cal = apply_calibrator(calibrator_pre, cal_method_pre, pre_test_prob_uncal)
pre_test_preds = (pre_test_prob_cal >= best_threshold_pretrain).astype(int)
# Prediksi Regresi (Hanya pada sampel hujan)
pre_test_preds_r_pre = np.zeros(len(y_te_o_pre))
rainy_te_pre = y_te_o_pre == 1
if reg_pretrain is not None and np.sum(rainy_te_pre) > 0:
    pre_test_preds_r_pre[rainy_te_pre] = np.clip(np.expm1(reg_pretrain.predict(X_te_seq_pre[rainy_te_pre], verbose=0).flatten()), 0, None)
# Jalankan Evaluasi Klasifikasi Pre-Trained
clf_metrics_pre, meteo_metrics_pre = evaluator_pre.evaluate_classification(
    y_te_o_pre, pre_test_prob_uncal, pre_test_prob_cal, pre_test_preds
)
ece_pre, mce_pre = compute_ece_mce(y_te_o_pre, pre_test_prob_cal)
clf_metrics_pre['ECE'] = ece_pre
clf_metrics_pre['MCE'] = mce_pre
evaluator_pre.save_combined_metrics()
# Simpan Confusion Matrix Pre-Trained
evaluator_pre.plot_confusion_matrix(y_te_o_pre, pre_test_preds, filename='confusion_matrix.png', suffix='(Pre-Trained)')
# Jalankan Evaluasi Regresi Pre-Trained
if np.sum(rainy_te_pre) > 0:
    reg_metrics_pre = evaluator_pre.evaluate_regression(y_te_r_pre[rainy_te_pre], pre_test_preds_r_pre[rainy_te_pre])
else:
    reg_metrics_pre = {}
# Buat Laporan Evaluasi Pre-Trained
evaluator_pre.generate_report(len(X_tr_seq), len(X_v_seq), len(X_te_seq_pre), list(scaler.feature_names_in_), clf_metrics_pre, meteo_metrics_pre, reg_metrics_pre, "LSTM (Pre-Trained)")
logger.info("Evaluasi Model Pre-Trained Selesai!")

# --- Bab 2: Evaluasi Model Fine-Tuning ---
# Dalam bab ini, model LSTM yang telah disetel ulang (*Fine-Tuned*) secara *warm-start* menggunakan data stasiun bumi AWS dievaluasi kinerjanya pada data uji sekuensial AWS Bumi.
# Bab 2: Evaluasi Model Fine-Tuning (LSTM)
logger.info("Memulai Bab 2: Evaluasi Model Fine-Tuning...")
evaluator_ft = ModelEvaluator(BASE_OUTPUT_DIR, phase='finetuning')
# Load model fine-tuned terbaik
clf_best_ft = tf.keras.models.load_model(evaluator_ft.dirs['models_clf'] / 'best_lstm_occ_finetuned.keras')
reg_best_ft = tf.keras.models.load_model(evaluator_ft.dirs['models_reg'] / 'best_lstm_reg_finetuned.keras') if (evaluator_ft.dirs['models_reg'] / 'best_lstm_reg_finetuned.keras').exists() else None
calibrator_ft = joblib.load(evaluator_ft.dirs['models_cal'] / 'isotonic_calibrator_finetuned.pkl')
# Load threshold optimal fine-tuning
_thresh_file_ft = evaluator_ft.dirs['models_clf'] / 'best_threshold_finetuned.pkl'
best_threshold_finetuned = joblib.load(_thresh_file_ft) if _thresh_file_ft.exists() else 0.5
logger.info(f"Threshold klasifikasi fine-tuning: {best_threshold_finetuned:.2f}")
# Prediksi Klasifikasi Fine-Tuned
test_probs_uncal = clf_best_ft.predict(X_te_seq, verbose=0).flatten()
test_probs_cal = apply_calibrator(calibrator_ft, cal_method_ft, test_probs_uncal)
test_preds_occ = (test_probs_cal >= best_threshold_finetuned).astype(int)
# Prediksi Regresi Fine-Tuned (Hanya pada sampel hujan)
rainy_te = y_te_o == 1
test_preds_r = np.zeros(len(y_te_o))
if reg_best_ft is not None and np.sum(rainy_te) > 0:
    test_preds_r[rainy_te] = np.clip(np.expm1(reg_best_ft.predict(X_te_seq[rainy_te], verbose=0).flatten()), 0, None)
# Jalankan Evaluasi Klasifikasi Fine-Tuned
clf_metrics_ft, meteo_metrics_ft = evaluator_ft.evaluate_classification(
    y_te_o, test_probs_uncal, test_probs_cal, test_preds_occ
)
ece_ft, mce_ft = compute_ece_mce(y_te_o, test_probs_cal)
clf_metrics_ft['ECE'] = ece_ft
clf_metrics_ft['MCE'] = mce_ft
evaluator_ft.save_combined_metrics()
# Simpan Confusion Matrix Fine-Tuned
evaluator_ft.plot_confusion_matrix(y_te_o, test_preds_occ, filename='confusion_matrix.png', suffix='(Fine-Tuned)')
# Jalankan Evaluasi Regresi Fine-Tuned
if np.sum(rainy_te) > 0:
    reg_metrics_ft = evaluator_ft.evaluate_regression(y_te_r[rainy_te], test_preds_r[rainy_te])
else:
    reg_metrics_ft = {}
# Buat Laporan Evaluasi Fine-Tuned
evaluator_ft.generate_report(len(X_tr_seq), len(X_v_seq), len(X_te_seq), list(scaler.feature_names_in_), clf_metrics_ft, meteo_metrics_ft, reg_metrics_ft, "LSTM (Fine-Tuned)")
logger.info("Evaluasi Model Fine-Tuned Selesai!")

# --- Bab 3: Perbandingan Model Pretraining vs Fine-Tuning ---
# Dalam bab ini, dilakukan pembandingan kurva evaluasi dan prediksi curah hujan LSTM dari kedua model (Pre-Trained vs Fine-Tuned) terhadap data aktual AWS bumi secara langsung.
# Bab 3: Perbandingan Model (LSTM)
logger.info("Memulai Bab 3: Perbandingan Model...")
comp_dir = BASE_OUTPUT_DIR / 'comparison'
comp_dir.mkdir(parents=True, exist_ok=True)
# Hitung prediksi pre-trained pada data uji AWS untuk perbandingan yang setara
pre_test_prob_uncal_comp = clf_pretrain.predict(X_te_seq, verbose=0).flatten()
pre_test_prob_cal_comp = apply_calibrator(calibrator_pre, cal_method_pre, pre_test_prob_uncal_comp)
pre_test_preds_r_comp = np.zeros(len(y_te_o))
rainy_te = y_te_o == 1
if reg_pretrain is not None and np.sum(rainy_te) > 0:
    pre_test_preds_r_comp[rainy_te] = np.clip(np.expm1(reg_pretrain.predict(X_te_seq[rainy_te], verbose=0).flatten()), 0, None)
# 1. Perbandingan Prediksi Kuantitas Hujan (Timeseries 336 Jam)
plt.figure(figsize=(15, 6))
start_idx = 0
end_idx = min(336, len(y_te_r))
plt.plot(y_te_r[start_idx:end_idx], label='Aktual (AWS)', color='black', marker='o', linewidth=2.0, markersize=4)
plt.plot(pre_test_preds_r_comp[start_idx:end_idx], label='Pre-Trained', color='blue', linestyle='--', linewidth=1.8)
plt.plot(test_preds_r[start_idx:end_idx], label='Fine-Tuned', color='red', linestyle='-', linewidth=2.2)
plt.title('Perbandingan Prediksi Curah Hujan (336 Jam Pertama)')
plt.xlabel('Waktu (Jam)')
plt.ylabel('Curah Hujan (mm/jam)')
plt.legend()
plt.tight_layout()
plt.savefig(comp_dir / 'prediction_comparison.png', dpi=300)
plt.show()
plt.close()
# 2. Perbandingan ROC Curve
plt.figure(figsize=(10, 8))
fpr_pre, tpr_pre, _ = roc_curve(y_te_o, pre_test_prob_cal_comp)
fpr_post, tpr_post, _ = roc_curve(y_te_o, test_probs_cal)
plt.plot(fpr_pre, tpr_pre, label=f'Pre-Trained (AUC={auc(fpr_pre, tpr_pre):.4f})', color='blue', linestyle='--', linewidth=2.0)
plt.plot(fpr_post, tpr_post, label=f'Fine-Tuned (AUC={auc(fpr_post, tpr_post):.4f})', color='red', linestyle='-', linewidth=2.5)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Perbandingan ROC Curve - Pre-Trained vs Fine-Tuned')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(comp_dir / 'roc_comparison.png', dpi=300)
plt.show()
plt.close()
# 3. Perbandingan Precision-Recall Curve
plt.figure(figsize=(10, 8))
prec_pre, rec_pre, _ = precision_recall_curve(y_te_o, pre_test_prob_cal_comp)
prec_post, rec_post, _ = precision_recall_curve(y_te_o, test_probs_cal)
plt.plot(rec_pre, prec_pre, label=f'Pre-Trained (PR-AUC={auc(rec_pre, prec_pre):.4f})', color='blue', linestyle='--', linewidth=2.0)
plt.plot(rec_post, prec_post, label=f'Fine-Tuned (PR-AUC={auc(rec_post, prec_post):.4f})', color='red', linestyle='-', linewidth=2.5)
plt.title('Perbandingan Precision-Recall Curve - Pre-Trained vs Fine-Tuned')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig(comp_dir / 'precision_recall_comparison.png', dpi=300)
plt.show()
plt.close()
# 4. Perbandingan Reliability Diagram (Calibration Curve)
plt.figure(figsize=(10, 8))
prob_true_pre, prob_pred_pre = calibration_curve(y_te_o, pre_test_prob_cal_comp, n_bins=10)
prob_true_post, prob_pred_post = calibration_curve(y_te_o, test_probs_cal, n_bins=10)
plt.plot(prob_pred_pre, prob_true_pre, marker='o', label='Pre-Trained', color='blue', linestyle='--', linewidth=2.0)
plt.plot(prob_pred_post, prob_true_post, marker='s', label='Fine-Tuned', color='red', linestyle='-', linewidth=2.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Perbandingan Reliability Diagram - Pre-Trained vs Fine-Tuned')
plt.xlabel('Prediksi Probabilitas')
plt.ylabel('Fraksi Positif Aktual')
plt.legend()
plt.tight_layout()
plt.savefig(comp_dir / 'reliability_comparison.png', dpi=300)
plt.show()
plt.close()
# 5. Perbandingan Probability Distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(pre_test_prob_cal_comp, label='Pre-Trained', color='blue', linestyle='--', linewidth=2.0, fill=True, alpha=0.3)
sns.kdeplot(test_probs_cal, label='Fine-Tuned', color='red', linestyle='-', linewidth=2.2, fill=True, alpha=0.3)
plt.close()
logger.info("Perbandingan Model Selesai!")
