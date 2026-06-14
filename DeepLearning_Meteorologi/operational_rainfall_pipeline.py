import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path

# Scikit-Learn Metrics & Models
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, brier_score_loss, confusion_matrix, log_loss, mean_squared_error, 
    mean_absolute_error, r2_score, mean_absolute_percentage_error, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
import joblib

# SHAP
import shap

# XGBoost
import xgboost as xgb

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CONFIG = {
    'SEED': 42,
    'TIME_STEPS_LSTM': 24, # 72 hours history
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'TEST_SPLIT': 0.2,
    'VAL_SPLIT': 0.1,
    'XGB_PARAMS_CLF': {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'device': 'cuda'
    },
    'XGB_PARAMS_REG': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'device': 'cuda'
    }
}

RAIN_CLASSES = {
    0: "No Rain (0 mm)",
    1: "Light Rain (0.1-15 mm)",
    2: "Moderate Rain (15-30 mm)",
    3: "Heavy Rain (>30 mm)"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Seed set to {seed}")

def get_paths():
    cwd = Path.cwd()
    if '/kaggle' in str(cwd) or '\\kaggle' in str(cwd):
        data_path = Path("/kaggle/input/datasets/jerismeteo/open-meteo-data-kebumen/open_meteo_jerukagung/cuaca_jerukagung.csv")
        out_dir = Path("/kaggle/working/outputs")
    else:
        data_path = Path(r"D:\Github\Projek_Rainfall\Analisis_Meteorologi\open_meteo_jerukagung\cuaca_jerukagung.csv")
        out_dir = Path("outputs")
    return data_path, out_dir

def create_directories(out_dir):
    dirs = [
        'lstm/models', 'lstm/scalers', 'lstm/metrics', 'lstm/predictions', 'lstm/probabilities', 'lstm/confusion_matrix', 'lstm/feature_importance', 'lstm/logs', 'lstm/plots',
        'xgboost/models', 'xgboost/scalers', 'xgboost/metrics', 'xgboost/predictions', 'xgboost/probabilities', 'xgboost/confusion_matrix', 'xgboost/feature_importance', 'xgboost/logs', 'xgboost/plots',
        'ensemble/predictions', 'ensemble/probabilities', 'ensemble/metrics', 'ensemble/plots', 'ensemble/models'
    ]
    for d in dirs:
        os.makedirs(out_dir / d, exist_ok=True)
    logger.info("Operational directories created successfully.")

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
def load_data(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    if 'datetime' in df.columns: df = df.set_index('datetime')
    elif 'date' in df.columns: df = df.set_index('date')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
    df.index.name = 'date'
    
    column_mapping = {
        'rain': 'rain', 'temperature_2m': 'temperature_2m', 'relative_humidity_2m': 'relative_humidity_2m',
        'dew_point_2m': 'dew_point_2m', 'surface_pressure': 'surface_pressure',
        'wind_direction_10m': 'wind_direction_10m', 'wind_speed_10m': 'wind_speed_10m'
    }
    df = df.rename(columns=column_mapping).sort_index()
    return df

def feature_engineering(df):
    logger.info("Feature engineering...")
    if (df['rain'] < 0).sum() > 0: df.loc[df['rain'] < 0, 'rain'] = 0
    df = df.interpolate(method='linear').bfill().ffill()
    
    if 'wind_direction_10m' in df.columns and 'wind_speed_10m' in df.columns:
        wind_dir_rad = np.radians(df['wind_direction_10m'])
        df['wind_u'] = -df['wind_speed_10m'] * np.sin(wind_dir_rad)
        df['wind_v'] = -df['wind_speed_10m'] * np.cos(wind_dir_rad)
        df = df.drop(columns=['wind_direction_10m'])
        
    if 'temperature_2m' in df.columns and 'dew_point_2m' in df.columns:
        df['dewpoint_depression'] = df['temperature_2m'] - df['dew_point_2m']
        
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'rain_lag_{lag}'] = df['rain'].shift(lag)
        
    for w in [3, 6, 12, 24]:
        df[f'rain_roll_mean_{w}h'] = df['rain'].rolling(window=w, min_periods=1).mean()
        df[f'rain_roll_max_{w}h'] = df['rain'].rolling(window=w, min_periods=1).max()
        
    if 'surface_pressure' in df.columns:
        for w in [1, 3, 6]: df[f'pressure_change_{w}h'] = df['surface_pressure'].diff(w)
            
    df['hour'], df['dayofyear'] = df.index.hour, df.index.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df = df.drop(columns=['hour', 'dayofyear']).dropna()
    return df

def categorize(mm):
    if mm == 0: return 0
    elif 0.1 <= mm <= 15: return 1
    elif 15 < mm <= 30: return 2
    else: return 3

def preprocess_data(df):
    logger.info("Resampling to 3-hourly & creating target...")
    df_3h = df.resample('3h').agg({
        'rain': 'sum', **{c: 'mean' for c in df.columns if c != 'rain' and 'rain' not in c},
        **{c: 'sum' for c in df.columns if c != 'rain' and 'rain' in c}
    }).dropna()
    df_3h['target_rain_mm'] = df_3h['rain'].shift(-1)
    df_3h = df_3h.dropna()
    df_3h['target_class'] = df_3h['target_rain_mm'].apply(categorize)
    return df_3h

# ==============================================================================
# 3. XGBOOST PIPELINE
# ==============================================================================
def train_xgboost(X_train, y_train_reg, y_train_clf, out_dir):
    logger.info("Training XGBoost Pipeline (Regressor + Classifier)...")
    weights = class_weight.compute_sample_weight('balanced', y_train_clf)
    xgb_clf = xgb.XGBClassifier(**CONFIG['XGB_PARAMS_CLF'], random_state=CONFIG['SEED'])
    xgb_clf.fit(X_train, y_train_clf, sample_weight=weights)
    xgb_clf.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_clf.json'))
    
    xgb_reg = xgb.XGBRegressor(**CONFIG['XGB_PARAMS_REG'], random_state=CONFIG['SEED'])
    xgb_reg.fit(X_train, y_train_reg)
    xgb_reg.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_reg.json'))
    return xgb_clf, xgb_reg

def predict_xgboost(xgb_clf, xgb_reg, X):
    return xgb_clf.predict_proba(X), xgb_clf.predict(X), np.maximum(0, xgb_reg.predict(X))

# ==============================================================================
# 4. LSTM PIPELINE
# ==============================================================================
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma, self.alpha = gamma, alpha
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.math.pow(1 - y_pred, self.gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

def create_sequences(X, y_reg, y_clf, time_steps):
    Xs, yrs, ycs = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        yrs.append(y_reg.iloc[i + time_steps])
        ycs.append(y_clf.iloc[i + time_steps])
    return np.array(Xs), np.array(yrs), np.array(ycs)

def train_lstm(X_train, y_train_reg, y_train_clf, X_val, y_val_reg, y_val_clf, out_dir):
    logger.info("Training Multi-Task LSTM...")
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    out_reg = layers.Dense(1, activation='linear', name='reg_output')(x)
    out_clf = layers.Dense(4, activation='softmax', name='clf_output')(x)
    model = keras.Model(inputs=inputs, outputs=[out_reg, out_clf])
    
    y_train_clf_oh, y_val_clf_oh = keras.utils.to_categorical(y_train_clf, 4), keras.utils.to_categorical(y_val_clf, 4)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss={'reg_output': 'mse', 'clf_output': CategoricalFocalLoss(gamma=2.0)},
        loss_weights={'reg_output': 1.0, 'clf_output': 5.0},
        metrics={'clf_output': 'accuracy'}
    )
    
    chkpt_path = out_dir / 'lstm' / 'models' / 'lstm_mtl.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(chkpt_path), monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0)
    ]
    history = model.fit(
        X_train, {'reg_output': y_train_reg, 'clf_output': y_train_clf_oh},
        validation_data=(X_val, {'reg_output': y_val_reg, 'clf_output': y_val_clf_oh}),
        epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'], callbacks=callbacks, verbose=1
    )
    return model, history

def predict_lstm(model, X):
    preds = model.predict(X, verbose=0)
    return preds[1], np.argmax(preds[1], axis=1), np.maximum(0, preds[0].flatten())

# ==============================================================================
# 5. ENSEMBLE PIPELINE (STACKING)
# ==============================================================================
def train_ensemble(lstm_val_prob, lstm_val_reg, xgb_val_prob, xgb_val_reg, y_val_clf, y_val_reg, out_dir):
    logger.info("Training Ensemble Meta-Models...")
    X_meta_clf = np.hstack([lstm_val_prob, xgb_val_prob])
    meta_clf = LogisticRegression(max_iter=1000, random_state=CONFIG['SEED'])
    meta_clf.fit(X_meta_clf, y_val_clf)
    
    X_meta_reg = np.column_stack([lstm_val_reg, xgb_val_reg])
    meta_reg = Ridge(random_state=CONFIG['SEED'])
    meta_reg.fit(X_meta_reg, y_val_reg)
    
    joblib.dump(meta_clf, out_dir / 'ensemble' / 'models' / 'meta_clf.pkl')
    joblib.dump(meta_reg, out_dir / 'ensemble' / 'models' / 'meta_reg.pkl')
    return meta_clf, meta_reg

def predict_ensemble(meta_clf, meta_reg, lstm_prob, lstm_reg, xgb_prob, xgb_reg):
    X_meta_clf = np.hstack([lstm_prob, xgb_prob])
    y_pred_prob = meta_clf.predict_proba(X_meta_clf)
    X_meta_reg = np.column_stack([lstm_reg, xgb_reg])
    return y_pred_prob, meta_clf.predict(X_meta_clf), np.maximum(0, meta_reg.predict(X_meta_reg))

# ==============================================================================
# 6. PUBLICATION-READY EVALUATION & PLOTTING
# ==============================================================================
def compute_meteorological_metrics(y_true, y_pred, num_classes=4):
    met_metrics = {}
    for c in range(1, num_classes): # Exclude Class 0 (No rain)
        hits = np.sum((y_pred == c) & (y_true == c))
        misses = np.sum((y_pred != c) & (y_true == c))
        false_alarms = np.sum((y_pred == c) & (y_true != c))
        correct_negatives = np.sum((y_pred != c) & (y_true != c))
        
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
        
        total = hits + misses + false_alarms + correct_negatives
        hits_random = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else 0
        ets = (hits - hits_random) / (hits + misses + false_alarms - hits_random) if (hits + misses + false_alarms - hits_random) > 0 else 0
        hss = (2 * (hits * correct_negatives - misses * false_alarms)) / ((hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (false_alarms + correct_negatives)) if total > 0 else 0
        
        met_metrics[f'Class_{c}'] = {'CSI': csi, 'POD': pod, 'FAR': far, 'ETS': ets, 'HSS': hss}
    return met_metrics

def evaluate_model(y_true_clf, y_pred_clf, y_pred_prob, y_true_reg, y_pred_reg, model_name, out_dir):
    logger.info(f"Evaluating {model_name}...")
    
    # Regression
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    mape = mean_absolute_percentage_error(y_true_reg + 1e-5, y_pred_reg)
    nse = 1 - (np.sum((y_true_reg - y_pred_reg)**2) / np.sum((y_true_reg - np.mean(y_true_reg))**2))
    
    # KGE
    r_pearson = np.corrcoef(y_true_reg, y_pred_reg)[0,1]
    alpha_kge = np.std(y_pred_reg) / (np.std(y_true_reg) + 1e-5)
    beta_kge = np.mean(y_pred_reg) / (np.mean(y_true_reg) + 1e-5)
    kge = 1 - np.sqrt((r_pearson - 1)**2 + (alpha_kge - 1)**2 + (beta_kge - 1)**2)
    
    # Classification
    acc = accuracy_score(y_true_clf, y_pred_clf)
    bal_acc = balanced_accuracy_score(y_true_clf, y_pred_clf)
    prec = precision_score(y_true_clf, y_pred_clf, average='macro', zero_division=0)
    rec = recall_score(y_true_clf, y_pred_clf, average='macro', zero_division=0)
    f1 = f1_score(y_true_clf, y_pred_clf, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_true_clf, y_pred_prob, multi_class='ovr')
    logloss = log_loss(y_true_clf, y_pred_prob)
    
    met_metrics = compute_meteorological_metrics(y_true_clf, y_pred_clf)
    
    report = {
        'Regression': {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse, 'KGE': kge, 'MAPE': mape},
        'Classification': {'Accuracy': acc, 'Balanced_Accuracy': bal_acc, 'Precision_Macro': prec, 'Recall_Macro': rec, 'F1_Macro': f1, 'ROC_AUC': roc_auc, 'LogLoss': logloss},
        'Meteorological': met_metrics
    }
    
    json_path = out_dir / model_name.lower() / 'metrics' / 'metrics_report.json'
    with open(json_path, 'w') as f: json.dump(report, f, indent=4)
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_true_clf, y_pred_clf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=RAIN_CLASSES.values(), yticklabels=RAIN_CLASSES.values())
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class'); plt.xlabel('Predicted Class')
    plt.savefig(out_dir / model_name.lower() / 'plots' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Prediction vs Observation
    plt.figure(figsize=(8,6))
    plt.scatter(y_true_reg, y_pred_reg, alpha=0.5, color='blue')
    plt.plot([0, max(y_true_reg)], [0, max(y_true_reg)], 'r--')
    plt.xlabel('Observed Rainfall (mm)'); plt.ylabel('Predicted Rainfall (mm)')
    plt.title(f'Pred vs Obs - {model_name}')
    plt.savefig(out_dir / model_name.lower() / 'plots' / 'pred_vs_obs.png', dpi=300, bbox_inches='tight')
    plt.close()
    plot_advanced_metrics(y_true_clf, y_pred_prob, model_name, out_dir)

def plot_advanced_metrics(y_true_clf, y_pred_prob, model_name, out_dir):
    logger.info(f"Generating Advanced Plots for {model_name}...")
    plot_path = out_dir / model_name.lower() / 'plots'
    
    # 1. Probability Distribution
    plt.figure(figsize=(8,5))
    for i in range(4):
        sns.kdeplot(y_pred_prob[:, i], label=f'Class {i}', fill=True)
    plt.title('Probability Distribution')
    plt.legend()
    plt.savefig(plot_path / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC & PR Curves
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    y_true_bin = label_binarize(y_true_clf, classes=[0,1,2,3])
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.subplot(1,2,2)
    for i in range(4):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        plt.plot(rec, prec, label=f'Class {i}')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Calibration Curve
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(8,6))
    for i in range(1, 4):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_pred_prob[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(plot_path / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_shap_plots(xgb_clf, lstm_model, X_train_scaled, X_train_seq, feature_names, out_dir):
    logger.info("Generating SHAP Explanations...")
    # XGBoost SHAP
    explainer_xgb = shap.TreeExplainer(xgb_clf)
    shap_values_xgb = explainer_xgb.shap_values(X_train_scaled)
    plt.figure()
    shap.summary_plot(shap_values_xgb, X_train_scaled, feature_names=feature_names, show=False)
    plt.savefig(out_dir / 'xgboost' / 'plots' / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # LSTM SHAP (Subsampled due to computational cost)
    logger.info("Running LSTM DeepExplainer on K-Means subset...")
    background = shap.kmeans(X_train_seq, 20) # Max 20 clusters for background to avoid OOM
    explainer_lstm = shap.DeepExplainer((lstm_model.inputs[0], lstm_model.outputs[1]), background)
    shap_values_lstm = explainer_lstm.shap_values(X_train_seq[:50]) # Only explain first 50 sequences
    
    # Plot SHAP for Class 1 (or list of all classes)
    plt.figure()
    # DeepExplainer returns list of arrays for multi-class. We plot Class 3 (Heavy Rain) if available, else Class 1.
    target_class = 3 if len(shap_values_lstm) > 3 else 0
    # Reshape sequence SHAP: aggregate across time steps (axis 1) by sum
    shap_agg = np.sum(shap_values_lstm[target_class], axis=1) 
    shap.summary_plot(shap_agg, X_train_seq[:50, -1, :], feature_names=feature_names, show=False)
    plt.savefig(out_dir / 'lstm' / 'plots' / 'shap_summary_class3.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    seed_everything(CONFIG['SEED'])
    data_path, out_dir = get_paths()
    create_directories(out_dir)
    
    df_raw = load_data(data_path)
    df_feat = feature_engineering(df_raw)
    df_3h = preprocess_data(df_feat)
    
    target_reg = df_3h['target_rain_mm']
    target_clf = df_3h['target_class']
    features = df_3h.drop(columns=['target_rain_mm', 'target_class'])
    feature_names = features.columns.tolist()
    
    # Train/Val/Test Split
    n_total = len(df_3h)
    n_test = int(n_total * CONFIG['TEST_SPLIT'])
    n_val = int(n_total * CONFIG['VAL_SPLIT'])
    n_train = n_total - n_test - n_val
    
    X_train_df, y_train_reg, y_train_clf = features.iloc[:n_train], target_reg.iloc[:n_train], target_clf.iloc[:n_train]
    X_val_df, y_val_reg, y_val_clf = features.iloc[n_train:n_train+n_val], target_reg.iloc[n_train:n_train+n_val], target_clf.iloc[n_train:n_train+n_val]
    X_test_df, y_test_reg, y_test_clf = features.iloc[n_train+n_val:], target_reg.iloc[n_train+n_val:], target_clf.iloc[n_train+n_val:]
    
    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=feature_names, index=X_train_df.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_df), columns=feature_names, index=X_val_df.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=feature_names, index=X_test_df.index)
    
    joblib.dump(scaler, out_dir / 'lstm' / 'scalers' / 'X_scaler.pkl')
    joblib.dump(scaler, out_dir / 'xgboost' / 'scalers' / 'X_scaler.pkl')
    
    # ---------------- XGBOOST PIPELINE ----------------
    xgb_clf, xgb_reg = train_xgboost(X_train_scaled, y_train_reg, y_train_clf, out_dir)
    xgb_val_prob, _, xgb_val_reg = predict_xgboost(xgb_clf, xgb_reg, X_val_scaled)
    xgb_test_prob, xgb_test_class, xgb_test_reg = predict_xgboost(xgb_clf, xgb_reg, X_test_scaled)
    evaluate_model(y_test_clf, xgb_test_class, xgb_test_prob, y_test_reg, xgb_test_reg, "XGBoost", out_dir)
    
    # ---------------- LSTM PIPELINE ----------------
    ts = CONFIG['TIME_STEPS_LSTM']
    X_train_seq, y_train_reg_seq, y_train_clf_seq = create_sequences(X_train_scaled, y_train_reg, y_train_clf, ts)
    X_val_seq, y_val_reg_seq, y_val_clf_seq = create_sequences(X_val_scaled, y_val_reg, y_val_clf, ts)
    X_test_seq, y_test_reg_seq, y_test_clf_seq = create_sequences(X_test_scaled, y_test_reg, y_test_clf, ts)
    
    lstm_model, _ = train_lstm(X_train_seq, y_train_reg_seq, y_train_clf_seq, X_val_seq, y_val_reg_seq, y_val_clf_seq, out_dir)
    lstm_val_prob, _, lstm_val_reg = predict_lstm(lstm_model, X_val_seq)
    lstm_test_prob, lstm_test_class, lstm_test_reg = predict_lstm(lstm_model, X_test_seq)
    evaluate_model(y_test_clf_seq, lstm_test_class, lstm_test_prob, y_test_reg_seq, lstm_test_reg, "LSTM", out_dir)
    
    # ---------------- ENSEMBLE PIPELINE ----------------
    meta_clf, meta_reg = train_ensemble(lstm_val_prob, lstm_val_reg, xgb_val_prob[ts:], xgb_val_reg[ts:], y_val_clf_seq, y_val_reg_seq, out_dir)
    ens_test_prob, ens_test_class, ens_test_reg = predict_ensemble(meta_clf, meta_reg, lstm_test_prob, lstm_test_reg, xgb_test_prob[ts:], xgb_test_reg[ts:])
    evaluate_model(y_test_clf_seq, ens_test_class, ens_test_prob, y_test_reg_seq, ens_test_reg, "Ensemble", out_dir)
    
    # SHAP Generation (Warning: LSTM DeepExplainer takes a long time)
    generate_shap_plots(xgb_clf, lstm_model, X_train_scaled, X_train_seq, feature_names, out_dir)
    
    logger.info("Operational Pipeline & SHAP Analysis Completed.")

if __name__ == '__main__':
    main()
