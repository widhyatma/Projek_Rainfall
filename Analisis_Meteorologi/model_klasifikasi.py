import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    log_loss, classification_report
)
from sklearn.calibration import calibration_curve

# XGBoost
import xgboost as xgb

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CLASSES = {
    0: "Dry (0 mm)",
    1: "Light (0.1 - 2.5 mm)",
    2: "Moderate (2.5 - 10 mm)",
    3: "Heavy (> 10 mm)"
}

def load_data(filepath):
    """Loads CSV data and adjusts timezone."""
    logger.info(f"Loading data from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
        
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
    df = df.sort_index()
    return df

def validate_meteorological_data(df):
    """Performs physical consistency checks."""
    logger.info("Performing meteorological consistency checks...")
    
    # 1. No negative rainfall
    neg_rain = (df['rain_mm'] < 0).sum()
    if neg_rain > 0:
        logger.warning(f"Found {neg_rain} negative rainfall values. Clipping to 0.")
        df.loc[df['rain_mm'] < 0, 'rain_mm'] = 0
        
    # 2. Time continuity
    # Check if there are gaps (we expect 3h gaps for the aggregated dataset)
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='3h')
    missing_times = expected_index.difference(df.index)
    if len(missing_times) > 0:
        logger.warning(f"Found {len(missing_times)} missing time steps. Reindexing and interpolating...")
        df = df.reindex(expected_index)
        
    # 3. Handle NaNs
    if df.isna().sum().sum() > 0:
        df['rain_mm'] = df['rain_mm'].fillna(0)
        df = df.interpolate(method='linear').bfill().ffill()
        
    # 4. Extreme events (> 50 mm / 3h)
    extreme_events = df[df['rain_mm'] > 50]
    if len(extreme_events) > 0:
        logger.warning(f"Flagged {len(extreme_events)} extreme rainfall events (>50mm/3h).")
        
    # Climatology check
    rain_freq = (df['rain_mm'] > 0).mean()
    logger.info(f"Climatological rain frequency: {rain_freq:.2%}")
    
    return df

def preprocess(df):
    """Generates features and aggregates to 3-hourly."""
    logger.info("Generating hourly features (lags, rolling stats, cyclical time)...")
    
    # 1. Rainfall Lags (1 to 24 hours)
    for i in range(1, 25):
        df[f'rain_lag_{i}'] = df['rain_mm'].shift(i)
        
    # 2. Rolling Statistics (Hourly)
    for window in [3, 6, 12, 24]:
        df[f'rain_roll_mean_{window}h'] = df['rain_mm'].rolling(window=window, min_periods=1).mean()
    for window in [3, 6, 12]:
        df[f'rain_roll_max_{window}h'] = df['rain_mm'].rolling(window=window, min_periods=1).max()
        
    # 3. Cyclical Time Features
    hours_in_day = 24
    days_in_year = 365.2425
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / hours_in_day)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / hours_in_day)
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / days_in_year)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / days_in_year)
    
    # Drop rows with NaNs from shifting
    df = df.dropna()
    
    logger.info("Resampling dataset to 3-hour aggregation...")
    
    # We want to predict the 3-hour accumulated rainfall for the current window.
    # Therefore, features (which represent history up to this point) should be taken at the end of the window (last),
    # while the target rain_mm is the sum of the current 3-hour window.
    agg_dict = {col: 'last' for col in df.columns}
    agg_dict['rain_mm'] = 'sum'
    
    df_3h = df.resample('3h').agg(agg_dict)
    
    # Validate
    df_3h = validate_meteorological_data(df_3h)
    
    # Target definition
    def categorize(mm):
        if mm < 0.1: return 0
        elif mm <= 2.5: return 1
        elif mm <= 10.0: return 2
        else: return 3
        
    df_3h['target_class'] = df_3h['rain_mm'].apply(categorize)
    return df_3h

def create_windows(X, y, time_steps=24):
    """Creates sequential sliding windows for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def expected_calibration_error(y_true_binary, y_prob_binary, n_bins=10):
    """Calculates the Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_binary, n_bins=n_bins)
    
    # Calculate bin counts to weight the ECE
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob_binary, bins) - 1
    
    bin_total = np.bincount(binids, minlength=len(bins))
    
    # We only care about populated bins
    ece = 0
    total_samples = len(y_prob_binary)
    for i in range(len(prob_pred)):
        # find matching bin using prob_pred (which is the mean of predictions in that bin)
        # approx method:
        idx = np.abs(bins[:-1] + np.diff(bins)/2 - prob_pred[i]).argmin()
        count = bin_total[idx]
        ece += (count / total_samples) * np.abs(prob_pred[i] - prob_true[i])
        
    return ece

def evaluate_model(model_name, y_true, y_prob):
    """Evaluates the model on Binary and Multi-class probabilistic metrics."""
    logger.info(f"Evaluating {model_name}...")
    
    # Binary Targets
    y_true_bin = (y_true > 0).astype(int)
    y_prob_bin = y_prob[:, 1:].sum(axis=1)
    y_pred_bin = (y_prob_bin > 0.5).astype(int)
    
    # Multi-class Targets
    y_pred = np.argmax(y_prob, axis=1)
    
    metrics = {}
    
    # Binary Metrics
    metrics['Binary F1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['Binary ROC-AUC'] = roc_auc_score(y_true_bin, y_prob_bin)
    metrics['Brier Score'] = brier_score_loss(y_true_bin, y_prob_bin)
    metrics['Binary Accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
    
    # Multi-class Metrics
    metrics['Macro F1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['Weighted F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Log Loss'] = log_loss(y_true, y_prob)
    
    # Calibration
    metrics['ECE'] = expected_calibration_error(y_true_bin, y_prob_bin)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[CLASSES[i] for i in range(4)], 
                yticklabels=[CLASSES[i] for i in range(4)])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # Plot Reliability Diagram
    prob_true, prob_pred = calibration_curve(y_true_bin, y_prob_bin, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Rain)')
    plt.title(f'Reliability Diagram: {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_calibration_curve.png')
    plt.close()
    
    return metrics

def compare_models(metrics_dict):
    """Outputs a markdown comparison table."""
    metrics_to_compare = ['Binary F1', 'Binary ROC-AUC', 'Macro F1', 'Brier Score', 'Log Loss', 'ECE']
    
    logger.info("\\n================ MODEL COMPARISON ================")
    header = "| Metric | XGBoost | LSTM | Winner |"
    divider = "|---|---|---|---|"
    
    print(header)
    print(divider)
    
    for m in metrics_to_compare:
        xgb_val = metrics_dict['XGBoost'][m]
        lstm_val = metrics_dict['LSTM'][m]
        
        # Lower is better for Brier, Log Loss, and ECE
        if m in ['Brier Score', 'Log Loss', 'ECE']:
            winner = 'XGBoost' if xgb_val < lstm_val else 'LSTM'
        else:
            winner = 'XGBoost' if xgb_val > lstm_val else 'LSTM'
            
        print(f"| {m} | {xgb_val:.4f} | {lstm_val:.4f} | **{winner}** |")
    
    print("==================================================")

def build_xgboost(X_train, y_train):
    """Trains a multi-class XGBoost model with class weighting."""
    logger.info("Training XGBoost Multi-Class Model...")
    
    # Compute class weights
    weights = class_weight.compute_sample_weight('balanced', y_train)
    
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    clf.fit(X_train, y_train, sample_weight=weights)
    return clf

def build_lstm(X_train, y_train):
    """Trains a sequential LSTM model."""
    logger.info("Training LSTM Sequence Model...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # We use a 10% validation split for early stopping
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model

def main():
    cwd = os.getcwd()
    filepath = os.path.join(cwd, "Analisis_Meteorologi", "open_meteo_jerukagung", "cuaca_jerukagung.csv")
    
    # 1. Load and Preprocess
    df = load_data(filepath)
    df_3h = preprocess(df)
    
    # Limit data for computational feasibility (e.g. 2017 onwards)
    df_3h = df_3h.loc['2005':].copy()
    
    # Exclude target from features
    feature_cols = [c for c in df_3h.columns if c not in ['rain_mm', 'target_class']]
    X_raw = df_3h[feature_cols].values
    y_raw = df_3h['target_class'].values
    
    # Scale Features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # --- XGBoost Data Prep (Tabular) ---
    # XGBoost predicts the current target using current row features.
    # We shift X by 1 so we use past data to predict the next 3h block target.
    X_xgb = X_scaled[:-1]
    y_xgb = y_raw[1:]
    
    split_xgb = int(len(X_xgb) * 0.8)
    X_train_xgb, y_train_xgb = X_xgb[:split_xgb], y_xgb[:split_xgb]
    X_test_xgb, y_test_xgb = X_xgb[split_xgb:], y_xgb[split_xgb:]
    
    # --- LSTM Data Prep (Sequential) ---
    TIME_STEPS = 24
    X_seq, y_seq = create_windows(X_scaled, y_raw, TIME_STEPS)
    
    split_seq = int(len(X_seq) * 0.8)
    X_train_seq, y_train_seq = X_seq[:split_seq], y_seq[:split_seq]
    X_test_seq, y_test_seq = X_seq[split_seq:], y_seq[split_seq:]
    
    # 2. Train Models
    xgb_model = build_xgboost(X_train_xgb, y_train_xgb)
    xgb_model.save_model(os.path.join(cwd, "xgboost_model.json"))
    
    lstm_model = build_lstm(X_train_seq, y_train_seq)
    lstm_model.save(os.path.join(cwd, "lstm_model.h5"))
    
    # 3. Evaluate Models
    logger.info("Generating Predictions...")
    xgb_prob = xgb_model.predict_proba(X_test_xgb)
    lstm_prob = lstm_model.predict(X_test_seq)
    
    # Since LSTM uses a sequence window, its test set starts TIME_STEPS-1 steps later than XGBoost.
    # The evaluation metrics handle this correctly as they pair true/pred directly.
    metrics_all = {
        'XGBoost': evaluate_model('XGBoost', y_test_xgb, xgb_prob),
        'LSTM': evaluate_model('LSTM', y_test_seq, lstm_prob)
    }
    
    # 4. Compare Models
    compare_models(metrics_all)

if __name__ == "__main__":
    main()
