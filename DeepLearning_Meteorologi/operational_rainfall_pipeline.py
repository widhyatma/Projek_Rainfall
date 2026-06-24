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
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, brier_score_loss, confusion_matrix, log_loss, mean_squared_error, 
    mean_absolute_error, r2_score, mean_absolute_percentage_error, 
    precision_recall_curve, roc_curve, auc, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import Ridge

# Ensembles
import xgboost as xgb
import lightgbm as lgb
import optuna

# SHAP
import shap

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Kaggle / TF Optimizations
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'SEED': 42,
    'TIME_STEPS_LSTM': 24, # 72 hours history
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'RUN_OPTUNA': False,  # Switch to true to run Bayesian Optimization
    'OPTUNA_TRIALS': 20
}

RAIN_CLASSES = {
    0: "No Rain (<0.5 mm)",
    1: "Light Rain (0.5-5 mm)",
    2: "Moderate Rain (5-20 mm)",
    3: "Heavy Rain (>20 mm)"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
        'xgboost/models', 'xgboost/metrics', 'xgboost/plots', 'xgboost/reports', 'xgboost/predictions',
        'lstm/models', 'lstm/metrics', 'lstm/plots', 'lstm/reports', 'lstm/predictions',
        'ensemble/models', 'ensemble/metrics', 'ensemble/plots', 'ensemble/reports', 'ensemble/predictions',
        'feature_analysis'
    ]
    for d in dirs:
        os.makedirs(out_dir / d, exist_ok=True)

# ==============================================================================
# FEATURE ENGINEERING & DATA SPLITTING
# ==============================================================================
def load_and_engineer_features(filepath):
    logger.info("Loading and Engineering Features...")
    df = pd.read_csv(filepath)
    if 'datetime' in df.columns: df = df.set_index('datetime')
    elif 'date' in df.columns: df = df.set_index('date')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Jakarta').tz_localize(None)
    df = df.sort_index()

    col_map = {'rain': 'rain', 'temperature_2m': 'temp', 'relative_humidity_2m': 'humidity', 'dew_point_2m': 'dew_point', 'surface_pressure': 'pressure'}
    df = df.rename(columns=lambda x: col_map.get(x, x))
    
    if (df['rain'] < 0).sum() > 0: df.loc[df['rain'] < 0, 'rain'] = 0
    df = df.interpolate(method='linear').bfill().ffill()

    # Dew point depression
    if 'temp' in df.columns and 'dew_point' in df.columns:
        df['dewpoint_depression'] = df['temp'] - df['dew_point']

    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        df[f'rain_lag_{lag}'] = df['rain'].shift(lag)
        if 'humidity' in df.columns: df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        if 'pressure' in df.columns and lag <= 6: df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag)

    # Trend features
    for w in [1, 3, 6]:
        if 'temp' in df.columns: df[f'temp_change_{w}h'] = df['temp'].diff(w)
        if 'humidity' in df.columns and w <= 3: df[f'humidity_change_{w}h'] = df['humidity'].diff(w)
        if 'pressure' in df.columns and w >= 3: df[f'pressure_change_{w}h'] = df['pressure'].diff(w)

    # Rolling statistics
    for w in [3, 6, 12, 24]:
        for c in ['rain', 'humidity', 'temp', 'pressure']:
            if c in df.columns:
                df[f'{c}_roll_mean_{w}h'] = df[c].rolling(window=w, min_periods=1).mean()
                df[f'{c}_roll_std_{w}h'] = df[c].rolling(window=w, min_periods=1).std().fillna(0)
                df[f'{c}_roll_min_{w}h'] = df[c].rolling(window=w, min_periods=1).min()
                df[f'{c}_roll_max_{w}h'] = df[c].rolling(window=w, min_periods=1).max()

    # Cyclical Time
    df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['sin_doy'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12.0)

    df = df.dropna()

    # Resample to 3-hourly forecasting blocks
    df_3h = df.resample('3h').agg({
        'rain': 'sum', **{c: 'mean' for c in df.columns if c != 'rain' and 'rain' not in c},
        **{c: 'sum' for c in df.columns if c != 'rain' and 'rain' in c}
    }).dropna()

    df_3h['target_rain_mm'] = df_3h['rain'].shift(-1)
    df_3h = df_3h.dropna()
    
    df_3h['target_occ'] = (df_3h['target_rain_mm'] >= 0.5).astype(int)
    
    def cat(mm):
        if mm < 0.5: return 0
        elif 0.5 <= mm <= 5: return 1
        elif 5 < mm <= 20: return 2
        else: return 3
    df_3h['target_cat'] = df_3h['target_rain_mm'].apply(cat)

    return df_3h

def split_chronological(df):
    logger.info("Splitting Data Chronologically...")
    train_mask = (df.index.year >= 2005) & (df.index.year <= 2023)
    val_mask = (df.index.year == 2024)
    test_mask = (df.index.year == 2025)
    
    # If the user has older/newer data, just default split
    if df[train_mask].empty: train_mask = df.index < df.index[int(len(df)*0.7)]
    if df[val_mask].empty: val_mask = (df.index >= df.index[int(len(df)*0.7)]) & (df.index < df.index[int(len(df)*0.85)])
    if df[test_mask].empty: test_mask = df.index >= df.index[int(len(df)*0.85)]
        
    return df[train_mask], df[val_mask], df[test_mask]

# ==============================================================================
# XGBOOST & OPTUNA MODULE
# ==============================================================================
def train_xgboost(X_train, y_train_occ, y_train_cat, y_train_reg, X_val, y_val_occ, y_val_cat, y_val_reg, out_dir):
    logger.info("Training XGBoost Pipeline...")
    
    xgb_params = {'tree_method': 'hist', 'random_state': CONFIG['SEED'], 'learning_rate': 0.05, 'n_estimators': 200}
    
    weights_cat = class_weight.compute_sample_weight('balanced', y_train_cat)
    
    clf_occ = xgb.XGBClassifier(**xgb_params, objective='binary:logistic')
    clf_occ.fit(X_train, y_train_occ, eval_set=[(X_val, y_val_occ)], verbose=0)
    
    clf_cat = xgb.XGBClassifier(**xgb_params, objective='multi:softprob', num_class=4)
    clf_cat.fit(X_train, y_train_cat, sample_weight=weights_cat, eval_set=[(X_val, y_val_cat)], verbose=0)
    
    reg_amt = xgb.XGBRegressor(**xgb_params, objective='reg:squarederror')
    reg_amt.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], verbose=0)
    
    clf_occ.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_occ.json'))
    clf_cat.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_cat.json'))
    reg_amt.save_model(str(out_dir / 'xgboost' / 'models' / 'xgb_reg.json'))
    
    return clf_occ, clf_cat, reg_amt

def predict_xgboost(clf_occ, clf_cat, reg_amt, X):
    occ_prob = clf_occ.predict_proba(X)[:, 1]
    cat_prob = clf_cat.predict_proba(X)
    amt_pred = np.maximum(0, reg_amt.predict(X))
    return occ_prob, cat_prob, amt_pred

# ==============================================================================
# KENDALL MTL LSTM MODULE
# ==============================================================================
class KendallMTLLoss(keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        # Log variances for uncertainty weighting
        self.log_var_occ = tf.Variable(0.0, trainable=True, dtype=tf.float32, name='log_var_occ')
        self.log_var_cat = tf.Variable(0.0, trainable=True, dtype=tf.float32, name='log_var_cat')
        self.log_var_reg = tf.Variable(0.0, trainable=True, dtype=tf.float32, name='log_var_reg')
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.occ_loss_tracker = keras.metrics.Mean(name="occ_loss")
        self.cat_loss_tracker = keras.metrics.Mean(name="cat_loss")
        self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")

    def call(self, inputs):
        return self.base_model(inputs)

    @tf.function
    def train_step(self, data):
        x, y = data
        y_occ, y_cat_oh, y_reg = y['out_occ'], y['out_cat'], y['out_reg']
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            pred_occ, pred_cat, pred_reg = y_pred[0], y_pred[1], y_pred[2]
            
            # Focal Loss for Binary Occurence
            bce = keras.losses.binary_crossentropy(y_occ, pred_occ)
            pt = tf.where(tf.equal(y_occ, 1), pred_occ, 1 - pred_occ)
            loss_occ = tf.reduce_mean(0.25 * tf.pow(1.0 - pt, 2.0) * bce)
            
            # Categorical Crossentropy
            loss_cat = tf.reduce_mean(keras.losses.categorical_crossentropy(y_cat_oh, pred_cat))
            
            # MSE
            loss_reg = tf.reduce_mean(keras.losses.mean_squared_error(y_reg, pred_reg))
            
            # Kendall Uncertainty Weighting
            w_occ = tf.exp(-self.log_var_occ) * loss_occ + self.log_var_occ
            w_cat = tf.exp(-self.log_var_cat) * loss_cat + self.log_var_cat
            w_reg = tf.exp(-self.log_var_reg) * loss_reg + self.log_var_reg
            
            total_loss = w_occ + w_cat + w_reg
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(total_loss)
        self.occ_loss_tracker.update_state(loss_occ)
        self.cat_loss_tracker.update_state(loss_cat)
        self.reg_loss_tracker.update_state(loss_reg)
        
        return {"loss": self.loss_tracker.result(), "occ": self.occ_loss_tracker.result(), "cat": self.cat_loss_tracker.result(), "reg": self.reg_loss_tracker.result()}
    
    @tf.function
    def test_step(self, data):
        x, y = data
        y_occ, y_cat_oh, y_reg = y['out_occ'], y['out_cat'], y['out_reg']
        y_pred = self(x, training=False)
        pred_occ, pred_cat, pred_reg = y_pred[0], y_pred[1], y_pred[2]
        
        loss_occ = tf.reduce_mean(keras.losses.binary_crossentropy(y_occ, pred_occ))
        loss_cat = tf.reduce_mean(keras.losses.categorical_crossentropy(y_cat_oh, pred_cat))
        loss_reg = tf.reduce_mean(keras.losses.mean_squared_error(y_reg, pred_reg))
        total_loss = tf.exp(-self.log_var_occ)*loss_occ + tf.exp(-self.log_var_cat)*loss_cat + tf.exp(-self.log_var_reg)*loss_reg
        
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}
        
    @property
    def metrics(self):
        return [self.loss_tracker, self.occ_loss_tracker, self.cat_loss_tracker, self.reg_loss_tracker]

def build_lstm(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    out_occ = layers.Dense(1, activation='sigmoid', name='out_occ')(x)
    out_cat = layers.Dense(4, activation='softmax', name='out_cat')(x)
    out_reg = layers.Dense(1, activation='linear', name='out_reg')(x)
    
    return keras.Model(inputs=inputs, outputs=[out_occ, out_cat, out_reg])

def create_sequences(X, y_occ, y_cat, y_reg, time_steps):
    Xs, yo, yc, yr = [], [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        yo.append(y_occ.iloc[i + time_steps])
        yc.append(y_cat.iloc[i + time_steps])
        yr.append(y_reg.iloc[i + time_steps])
    return np.array(Xs), np.array(yo), np.array(yc), np.array(yr)

# ==============================================================================
# ENSEMBLE STACKING MODULE
# ==============================================================================
def train_ensemble(xgb_occ, xgb_cat, xgb_reg, lstm_occ, lstm_cat, lstm_reg, y_occ, y_cat, y_reg, out_dir):
    logger.info("Training Meta-Models (LightGBM & Ridge)...")
    
    # Binary Meta
    X_meta_occ = np.column_stack([xgb_occ, lstm_occ])
    meta_occ = lgb.LGBMClassifier(random_state=CONFIG['SEED'], n_estimators=50, verbose=-1)
    meta_occ.fit(X_meta_occ, y_occ)
    
    # Multiclass Meta
    X_meta_cat = np.hstack([xgb_cat, lstm_cat])
    meta_cat = lgb.LGBMClassifier(random_state=CONFIG['SEED'], n_estimators=50, verbose=-1)
    meta_cat.fit(X_meta_cat, y_cat)
    
    # Regression Meta
    X_meta_reg = np.column_stack([xgb_reg, lstm_reg])
    meta_reg = Ridge(random_state=CONFIG['SEED'])
    meta_reg.fit(X_meta_reg, y_reg)
    
    import joblib
    joblib.dump(meta_occ, out_dir / 'ensemble' / 'models' / 'meta_occ.pkl')
    joblib.dump(meta_cat, out_dir / 'ensemble' / 'models' / 'meta_cat.pkl')
    joblib.dump(meta_reg, out_dir / 'ensemble' / 'models' / 'meta_reg.pkl')
    
    return meta_occ, meta_cat, meta_reg

# ==============================================================================
# EVALUATION & PLOTS
# ==============================================================================
def compute_meteorological_metrics(y_true, y_pred, num_classes=4):
    met = {}
    for c in range(1, num_classes):
        hits = np.sum((y_pred == c) & (y_true == c))
        misses = np.sum((y_pred != c) & (y_true == c))
        fa = np.sum((y_pred == c) & (y_true != c))
        cn = np.sum((y_pred != c) & (y_true != c))
        
        pod = hits / (hits + misses + 1e-7)
        far = fa / (hits + fa + 1e-7)
        csi = hits / (hits + misses + fa + 1e-7)
        
        tot = hits + misses + fa + cn
        hits_rand = ((hits + misses)*(hits + fa)) / (tot + 1e-7)
        ets = (hits - hits_rand) / (hits + misses + fa - hits_rand + 1e-7)
        hss = 2*(hits*cn - misses*fa) / ((hits+misses)*(misses+cn) + (hits+fa)*(fa+cn) + 1e-7)
        
        met[f'Class_{c}'] = {'POD': pod, 'FAR': far, 'CSI': csi, 'ETS': ets, 'HSS': hss}
    return met

def evaluate_pipeline(y_occ, p_occ, y_cat, p_cat_prob, y_reg, p_reg, name, out_dir):
    logger.info(f"Evaluating {name}...")
    p_cat = np.argmax(p_cat_prob, axis=1)
    p_occ_bin = (p_occ > 0.5).astype(int)
    
    # Binary
    bin_acc = accuracy_score(y_occ, p_occ_bin)
    bin_f1 = f1_score(y_occ, p_occ_bin, zero_division=0)
    bin_brier = brier_score_loss(y_occ, p_occ)
    
    # Multiclass
    cat_acc = accuracy_score(y_cat, p_cat)
    cat_f1_mac = f1_score(y_cat, p_cat, average='macro', zero_division=0)
    cat_kappa = cohen_kappa_score(y_cat, p_cat)
    met_metrics = compute_meteorological_metrics(y_cat, p_cat)
    
    # Regression
    rmse = np.sqrt(mean_squared_error(y_reg, p_reg))
    mae = mean_absolute_error(y_reg, p_reg)
    r2 = r2_score(y_reg, p_reg)
    nse = 1 - (np.sum((y_reg - p_reg)**2) / (np.sum((y_reg - np.mean(y_reg))**2) + 1e-7))
    
    rep = {
        'Binary': {'Acc': bin_acc, 'F1': bin_f1, 'Brier': bin_brier},
        'Multiclass': {'Acc': cat_acc, 'MacroF1': cat_f1_mac, 'Kappa': cat_kappa},
        'Meteorological': met_metrics,
        'Regression': {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse}
    }
    
    with open(out_dir / name.lower() / 'metrics' / 'report.json', 'w') as f:
        json.dump(rep, f, indent=4)
        
    # Plot 1: Pred vs Obs Scatter
    plt.figure()
    plt.scatter(y_reg, p_reg, alpha=0.5)
    plt.plot([0, max(y_reg)], [0, max(y_reg)], 'r--')
    plt.xlabel("Observed (mm)"); plt.ylabel("Predicted (mm)")
    plt.savefig(out_dir / name.lower() / 'plots' / 'pred_vs_obs.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    seed_everything(CONFIG['SEED'])
    data_path, out_dir = get_paths()
    create_directories(out_dir)
    
    df_raw = load_and_engineer_features(data_path)
    train_df, val_df, test_df = split_chronological(df_raw)
    
    cols_drop = ['target_rain_mm', 'target_occ', 'target_cat']
    feat_cols = train_df.drop(columns=cols_drop).columns.tolist()
    
    scaler = MinMaxScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train_df.drop(columns=cols_drop)), columns=feat_cols, index=train_df.index)
    X_va = pd.DataFrame(scaler.transform(val_df.drop(columns=cols_drop)), columns=feat_cols, index=val_df.index)
    X_te = pd.DataFrame(scaler.transform(test_df.drop(columns=cols_drop)), columns=feat_cols, index=test_df.index)
    
    # 1. XGBoost
    xgb_occ, xgb_cat, xgb_reg = train_xgboost(
        X_tr, train_df['target_occ'], train_df['target_cat'], train_df['target_rain_mm'],
        X_va, val_df['target_occ'], val_df['target_cat'], val_df['target_rain_mm'], out_dir
    )
    
    xo_te, xc_te, xr_te = predict_xgboost(xgb_occ, xgb_cat, xgb_reg, X_te)
    evaluate_pipeline(test_df['target_occ'], xo_te, test_df['target_cat'], xc_te, test_df['target_rain_mm'], xr_te, "XGBoost", out_dir)
    
    # 2. LSTM
    ts = CONFIG['TIME_STEPS_LSTM']
    X_tr_s, yo_tr, yc_tr, yr_tr = create_sequences(X_tr, train_df['target_occ'], train_df['target_cat'], train_df['target_rain_mm'], ts)
    X_va_s, yo_va, yc_va, yr_va = create_sequences(X_va, val_df['target_occ'], val_df['target_cat'], val_df['target_rain_mm'], ts)
    X_te_s, yo_te, yc_te, yr_te = create_sequences(X_te, test_df['target_occ'], test_df['target_cat'], test_df['target_rain_mm'], ts)
    
    yc_tr_oh = keras.utils.to_categorical(yc_tr, 4)
    yc_va_oh = keras.utils.to_categorical(yc_va, 4)
    
    base_lstm = build_lstm((ts, len(feat_cols)))
    mtl_model = KendallMTLLoss(base_lstm)
    mtl_model.compile(optimizer=keras.optimizers.Adam(1e-3))
    
    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    mtl_model.fit(
        X_tr_s, {'out_occ': yo_tr, 'out_cat': yc_tr_oh, 'out_reg': yr_tr},
        validation_data=(X_va_s, {'out_occ': yo_va, 'out_cat': yc_va_oh, 'out_reg': yr_va}),
        epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'], callbacks=callbacks, verbose=1
    )
    
    lstm_te = mtl_model.predict(X_te_s)
    evaluate_pipeline(yo_te, lstm_te[0].flatten(), yc_te, lstm_te[1], yr_te, lstm_te[2].flatten(), "LSTM", out_dir)

    # 3. Ensemble
    xo_va, xc_va, xr_va = predict_xgboost(xgb_occ, xgb_cat, xgb_reg, X_va)
    lstm_va = mtl_model.predict(X_va_s)
    
    meta_occ, meta_cat, meta_reg = train_ensemble(
        xo_va[ts:], xc_va[ts:], xr_va[ts:], lstm_va[0].flatten(), lstm_va[1], lstm_va[2].flatten(),
        yo_va, yc_va, yr_va, out_dir
    )
    
    eo_te = meta_occ.predict_proba(np.column_stack([xo_te[ts:], lstm_te[0].flatten()]))[:, 1]
    ec_te = meta_cat.predict_proba(np.hstack([xc_te[ts:], lstm_te[1]]))
    er_te = np.maximum(0, meta_reg.predict(np.column_stack([xr_te[ts:], lstm_te[2].flatten()])))
    
    evaluate_pipeline(yo_te, eo_te, yc_te, ec_te, yr_te, er_te, "Ensemble", out_dir)

    logger.info("Research-Grade Operational Pipeline Finished.")

if __name__ == '__main__':
    main()
