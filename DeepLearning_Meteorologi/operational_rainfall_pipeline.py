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
    log_loss, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
import joblib

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
    data_path = Path(r"D:\Github\Projek_Rainfall\Analisis_Meteorologi\open_meteo_jerukagung\cuaca_jerukagung.csv")
    out_dir = Path("outputs")
    return data_path, out_dir

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
def load_data(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
    df.index.name = 'date'
    
    column_mapping = {
        'rain': 'rain',
        'temperature_2m': 'temperature_2m',
        'relative_humidity_2m': 'relative_humidity_2m',
        'dew_point_2m': 'dew_point_2m',
        'surface_pressure': 'surface_pressure',
        'wind_direction_10m': 'wind_direction_10m',
        'wind_speed_10m': 'wind_speed_10m'
    }
    df = df.rename(columns=column_mapping)
    df = df.sort_index()
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
        for w in [1, 3, 6]:
            df[f'pressure_change_{w}h'] = df['surface_pressure'].diff(w)
            
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df = df.drop(columns=['hour', 'dayofyear'])
    
    df = df.dropna()
    return df

def categorize(mm):
    if mm == 0: return 0
    elif 0.1 <= mm <= 15: return 1
    elif 15 < mm <= 30: return 2
    else: return 3

def preprocess_data(df):
    logger.info("Resampling to 3-hourly & creating target...")
    df_3h = df.resample('3h').agg({
        'rain': 'sum',
        **{c: 'mean' for c in df.columns if c != 'rain' and 'rain' not in c},
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
    
    # Train Classifier
    weights = class_weight.compute_sample_weight('balanced', y_train_clf)
    xgb_clf = xgb.XGBClassifier(**CONFIG['XGB_PARAMS_CLF'], random_state=CONFIG['SEED'])
    xgb_clf.fit(X_train, y_train_clf, sample_weight=weights)
    xgb_clf.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_clf.json'))
    
    # Train Regressor
    xgb_reg = xgb.XGBRegressor(**CONFIG['XGB_PARAMS_REG'], random_state=CONFIG['SEED'])
    xgb_reg.fit(X_train, y_train_reg)
    xgb_reg.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_reg.json'))
    
    return xgb_clf, xgb_reg

def predict_xgboost(xgb_clf, xgb_reg, X):
    y_pred_prob = xgb_clf.predict_proba(X)
    y_pred_class = xgb_clf.predict(X)
    y_pred_reg = xgb_reg.predict(X)
    y_pred_reg = np.maximum(0, y_pred_reg) # Prevent negative rain
    return y_pred_prob, y_pred_class, y_pred_reg

# ==============================================================================
# 4. LSTM PIPELINE
# ==============================================================================
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.math.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

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
    
    # Head 1: Regression
    out_reg = layers.Dense(1, activation='linear', name='reg_output')(x)
    
    # Head 2: Classification
    out_clf = layers.Dense(4, activation='softmax', name='clf_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=[out_reg, out_clf])
    
    y_train_clf_onehot = keras.utils.to_categorical(y_train_clf, num_classes=4)
    y_val_clf_onehot = keras.utils.to_categorical(y_val_clf, num_classes=4)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss={'reg_output': 'mse', 'clf_output': CategoricalFocalLoss(gamma=2.0)},
        loss_weights={'reg_output': 1.0, 'clf_output': 5.0},
        metrics={'clf_output': 'accuracy'}
    )
    
    chkpt_path = out_dir / 'lstm' / 'models' / 'lstm_mtl.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(chkpt_path), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]
    
    history = model.fit(
        X_train, {'reg_output': y_train_reg, 'clf_output': y_train_clf_onehot},
        validation_data=(X_val, {'reg_output': y_val_reg, 'clf_output': y_val_clf_onehot}),
        epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
        callbacks=callbacks, verbose=1
    )
    return model, history

def predict_lstm(model, X):
    preds = model.predict(X)
    y_pred_reg = np.maximum(0, preds[0].flatten())
    y_pred_prob = preds[1]
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    return y_pred_prob, y_pred_class, y_pred_reg

# ==============================================================================
# 5. ENSEMBLE PIPELINE (STACKING)
# ==============================================================================
def train_ensemble(lstm_val_prob, lstm_val_reg, xgb_val_prob, xgb_val_reg, y_val_clf, y_val_reg, out_dir):
    logger.info("Training Ensemble Meta-Models...")
    
    # Meta-Classifier
    X_meta_clf = np.hstack([lstm_val_prob, xgb_val_prob])
    meta_clf = LogisticRegression(max_iter=1000, random_state=CONFIG['SEED'])
    meta_clf.fit(X_meta_clf, y_val_clf)
    
    # Meta-Regressor
    X_meta_reg = np.column_stack([lstm_val_reg, xgb_val_reg])
    meta_reg = Ridge(random_state=CONFIG['SEED'])
    meta_reg.fit(X_meta_reg, y_val_reg)
    
    joblib.dump(meta_clf, out_dir / 'ensemble' / 'models' / 'meta_clf.pkl')
    joblib.dump(meta_reg, out_dir / 'ensemble' / 'models' / 'meta_reg.pkl')
    
    return meta_clf, meta_reg

def predict_ensemble(meta_clf, meta_reg, lstm_prob, lstm_reg, xgb_prob, xgb_reg):
    X_meta_clf = np.hstack([lstm_prob, xgb_prob])
    y_pred_prob = meta_clf.predict_proba(X_meta_clf)
    y_pred_class = meta_clf.predict(X_meta_clf)
    
    X_meta_reg = np.column_stack([lstm_reg, xgb_reg])
    y_pred_reg = meta_reg.predict(X_meta_reg)
    y_pred_reg = np.maximum(0, y_pred_reg)
    
    return y_pred_prob, y_pred_class, y_pred_reg

# ==============================================================================
# 6. OPERATIONAL OUTPUT & METRICS
# ==============================================================================
def save_operational_output(timestamps, y_pred_reg, y_pred_prob, y_pred_class, model_name, out_dir):
    df_out = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_rainfall_mm': np.round(y_pred_reg, 2),
        'rain_probability': np.round((1.0 - y_pred_prob[:, 0]) * 100, 2),
        'rain_class': [RAIN_CLASSES[c] for c in y_pred_class],
        'model': model_name
    })
    df_out.to_csv(out_dir / model_name.lower() / 'predictions' / 'operational_forecast.csv', index=False)
    logger.info(f"Saved operational output for {model_name}")

def calculate_metrics(y_true_clf, y_pred_clf, y_true_reg, y_pred_reg, name):
    acc = accuracy_score(y_true_clf, y_pred_clf)
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    logger.info(f"{name} Metrics -> Accuracy: {acc:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    seed_everything(CONFIG['SEED'])
    data_path, out_dir = get_paths()
    
    df_raw = load_data(data_path)
    df_feat = feature_engineering(df_raw)
    df_3h = preprocess_data(df_feat)
    
    target_reg = df_3h['target_rain_mm']
    target_clf = df_3h['target_class']
    features = df_3h.drop(columns=['target_rain_mm', 'target_class'])
    
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
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns, index=X_train_df.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_df), columns=X_val_df.columns, index=X_val_df.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns, index=X_test_df.index)
    
    joblib.dump(scaler, out_dir / 'lstm' / 'scalers' / 'X_scaler.pkl')
    joblib.dump(scaler, out_dir / 'xgboost' / 'scalers' / 'X_scaler.pkl')
    
    # ---------------- XGBOOST PIPELINE ----------------
    xgb_clf, xgb_reg = train_xgboost(X_train_scaled, y_train_reg, y_train_clf, out_dir)
    xgb_val_prob, xgb_val_class, xgb_val_reg = predict_xgboost(xgb_clf, xgb_reg, X_val_scaled)
    xgb_test_prob, xgb_test_class, xgb_test_reg = predict_xgboost(xgb_clf, xgb_reg, X_test_scaled)
    
    calculate_metrics(y_test_clf, xgb_test_class, y_test_reg, xgb_test_reg, "XGBoost")
    save_operational_output(X_test_scaled.index, xgb_test_reg, xgb_test_prob, xgb_test_class, "xgboost", out_dir)
    
    # ---------------- LSTM PIPELINE ----------------
    ts = CONFIG['TIME_STEPS_LSTM']
    X_train_seq, y_train_reg_seq, y_train_clf_seq = create_sequences(X_train_scaled, y_train_reg, y_train_clf, ts)
    X_val_seq, y_val_reg_seq, y_val_clf_seq = create_sequences(X_val_scaled, y_val_reg, y_val_clf, ts)
    X_test_seq, y_test_reg_seq, y_test_clf_seq = create_sequences(X_test_scaled, y_test_reg, y_test_clf, ts)
    
    lstm_model, lstm_history = train_lstm(X_train_seq, y_train_reg_seq, y_train_clf_seq, X_val_seq, y_val_reg_seq, y_val_clf_seq, out_dir)
    lstm_val_prob, lstm_val_class, lstm_val_reg = predict_lstm(lstm_model, X_val_seq)
    lstm_test_prob, lstm_test_class, lstm_test_reg = predict_lstm(lstm_model, X_test_seq)
    
    calculate_metrics(y_test_clf_seq, lstm_test_class, y_test_reg_seq, lstm_test_reg, "LSTM")
    save_operational_output(X_test_scaled.index[ts:], lstm_test_reg, lstm_test_prob, lstm_test_class, "lstm", out_dir)
    
    # ---------------- ENSEMBLE PIPELINE ----------------
    # Align val lengths (LSTM loses first ts steps)
    meta_clf, meta_reg = train_ensemble(
        lstm_val_prob, lstm_val_reg, xgb_val_prob[ts:], xgb_val_reg[ts:], 
        y_val_clf_seq, y_val_reg_seq, out_dir
    )
    
    ens_test_prob, ens_test_class, ens_test_reg = predict_ensemble(
        meta_clf, meta_reg, lstm_test_prob, lstm_test_reg, xgb_test_prob[ts:], xgb_test_reg[ts:]
    )
    
    calculate_metrics(y_test_clf_seq, ens_test_class, y_test_reg_seq, ens_test_reg, "Ensemble")
    save_operational_output(X_test_scaled.index[ts:], ens_test_reg, ens_test_prob, ens_test_class, "ensemble", out_dir)
    
    logger.info("Pipeline execution completed.")

if __name__ == '__main__':
    main()
