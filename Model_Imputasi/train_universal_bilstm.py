#!/usr/bin/env python
"""
Universal Bi-LSTM Imputation Model Training Pipeline
====================================================
Trains the Universal Multi-Input Bi-LSTM on multi-node AWS data (id-03 and id-05)
and ERA5 reanalysis anchors.
Supports both SQLite Database (meteo_local_cache.db) and CSV cache inputs.
"""

import os
import sys
import glob
import sqlite3
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# Add framework package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framework.preprocessor import MeteorologicalPreprocessor, IS_KAGGLE, xp
from framework.network import build_universal_bilstm_model, mask_weighted_mse
from framework.evaluator import MeteorologicalEvaluator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as tf_callbacks


def parse_args():
    parser = argparse.ArgumentParser(description="Train Universal Bi-LSTM Imputation Network")
    parser.add_argument('--mode', type=str, default='FULL_TRAIN', choices=['LOCAL_TEST', 'FULL_TRAIN'],
                        help="Execution mode (LOCAL_TEST for quick debug, FULL_TRAIN for production)")
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="Training batch size")
    parser.add_argument('--lookback', type=int, default=120, help="Sliding window lookback horizon (minutes)")
    parser.add_argument('--nodes', type=str, default='id-03,id-05', help="Comma-separated station node IDs to train on")
    parser.add_argument('--db_path', type=str, default=r"D:\Github\Firebase_Database_Administrator\databases\meteo_local_cache.db",
                        help="Path to SQLite database")
    return parser.parse_args()


def load_node_data_from_db(db_path, target_nodes, target_vars, preprocessor, logger):
    """Loads station data directly from SQLite database."""
    logger.info(f"Connecting to SQLite Database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    node_data = {}
    node_name_map = {}
    col_map = {'dew_point': 'dewpoint', 'dew': 'dewpoint', 'temperature': 'temperature', 'humidity': 'humidity', 'pressure': 'pressure'}
    
    for idx, node_id in enumerate(sorted(target_nodes)):
        node_name_map[node_id] = idx
        logger.info(f"Querying measurements for {node_id} from database...")
        
        query = f"""
            SELECT unix_ts as timestamp, temperature, humidity, pressure, dew_point, dew
            FROM aws_measurements
            WHERE station_id = '{node_id}' AND unix_ts IS NOT NULL
            ORDER BY unix_ts ASC
        """
        df_raw = pd.read_sql_query(query, conn)
        
        if df_raw.empty:
            logger.warning(f"No records found in SQLite for {node_id}.")
            continue
            
        # Unify dewpoint column and keep only 1 dewpoint
        if 'dew_point' in df_raw.columns and 'dew' in df_raw.columns:
            df_raw['dewpoint'] = df_raw['dew_point'].combine_first(df_raw['dew'])
            df_raw = df_raw.drop(columns=['dew_point', 'dew'])
        elif 'dew_point' in df_raw.columns:
            df_raw['dewpoint'] = df_raw['dew_point']
            df_raw = df_raw.drop(columns=['dew_point'])
        elif 'dew' in df_raw.columns:
            df_raw['dewpoint'] = df_raw['dew']
            df_raw = df_raw.drop(columns=['dew'])

        df_raw['datetime'] = pd.to_datetime(df_raw['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
        df_raw = df_raw.dropna(subset=['datetime']).sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        
        df_renamed = df_raw.rename(columns=col_map)
        df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated()]
        df_qc = preprocessor.apply_qc(df_renamed)
        
        # Resample to continuous 1-minute grid
        df_1min = df_qc[[c for c in target_vars if c in df_qc.columns]].resample('1min').mean()
        node_data[node_id] = df_1min
        logger.info(f"[OK] Loaded {node_id} from SQLite: {len(df_1min):,} 1-minute records ({df_1min.index.min()} to {df_1min.index.max()})")
        
    conn.close()
    return node_data, node_name_map


def main():
    args = parse_args()
    target_nodes = [n.strip() for n in args.nodes.split(',') if n.strip()]
    
    # 1. Environment Detection & Directory Setup
    if IS_KAGGLE:
        print("[INFO] Running in Kaggle Cloud Environment")
        BASE_DIR = "/kaggle/working"
        CACHE_DIR = "/kaggle/input/notebooks/jerismeteo/cek-data-sensor"
        PATH_ERA5 = glob.glob('/kaggle/input/**/cuaca_gabungan_jerukagung.csv', recursive=True)
        PATH_ERA5 = PATH_ERA5[0] if PATH_ERA5 else '/kaggle/input/generate-era5-data/cuaca_gabungan_jerukagung.csv'
        DB_PATH = None
    else:
        print("[INFO] Running in Local Windows Environment")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR = os.path.join(os.path.dirname(BASE_DIR), "Analisis_Meteorologi", "cache_data")
        PATH_ERA5 = os.path.join(BASE_DIR, "cuaca_gabungan_jerukagung.csv")
        DB_PATH = args.db_path

    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    for sub in ['model', 'metrics', 'plots', 'reports', 'imputed_data', 'logs']:
        os.makedirs(os.path.join(OUTPUTS_DIR, sub), exist_ok=True)

    # 2. Configure Logging
    log_file = os.path.join(OUTPUTS_DIR, 'logs', f"train_{args.mode.lower()}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("UniversalTrainer")
    logger.info(f"Initialized Universal Training for Nodes: {target_nodes}. Mode: {args.mode}, Lookback: {args.lookback}")

    # Set Hyperparameters based on mode
    epochs = args.epochs or (6 if args.mode == 'LOCAL_TEST' else 100)
    batch_size = args.batch_size or (64 if args.mode == 'LOCAL_TEST' else 512)
    
    target_vars = ['temperature', 'humidity', 'pressure', 'dewpoint']
    col_map = {'dew': 'dewpoint', 'dew_point': 'dewpoint', 'temperature': 'temperature', 'humidity': 'humidity', 'pressure': 'pressure'}

    preprocessor = MeteorologicalPreprocessor(target_vars=target_vars, lookback=args.lookback)
    
    # 3. Load Data from SQLite DB if exists, else CSV cache
    node_data = {}
    node_name_map = {}
    
    if DB_PATH and os.path.exists(DB_PATH):
        logger.info(f"Found SQLite Database: {DB_PATH}")
        node_data, node_name_map = load_node_data_from_db(DB_PATH, target_nodes, target_vars, preprocessor, logger)
    
    if not node_data:
        logger.info(f"Loading from CSV cache at {CACHE_DIR}...")
        for idx, node_id in enumerate(sorted(target_nodes)):
            fpath = os.path.join(CACHE_DIR, f"{node_id}_raw.csv")
            if not os.path.exists(fpath):
                fpath = os.path.join(CACHE_DIR, f"{node_id}_5days_recent.csv")
            if not os.path.exists(fpath):
                logger.warning(f"File for {node_id} not found in {CACHE_DIR}. Skipping.")
                continue
                
            node_name_map[node_id] = idx
            df_raw = pd.read_csv(fpath)
            if 'timestamp' in df_raw.columns:
                df_raw['datetime'] = pd.to_datetime(df_raw['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
            elif 'datetime_wib' in df_raw.columns:
                df_raw['datetime'] = pd.to_datetime(df_raw['datetime_wib'])
            else:
                continue
                
            df_raw = df_raw.dropna(subset=['datetime']).sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
            df_renamed = df_raw.rename(columns=col_map)
            df_qc = preprocessor.apply_qc(df_renamed)
            df_1min = df_qc[[c for c in target_vars if c in df_qc.columns]].resample('1min').mean()
            node_data[node_id] = df_1min
            logger.info(f"Loaded {node_id} from CSV: {len(df_1min)} records")

    if not node_data:
        logger.error("No node data could be loaded. Aborting.")
        sys.exit(1)

    # 4. Load ERA5 Reanalysis Anchors if available
    era5_df = None
    if os.path.exists(PATH_ERA5):
        logger.info(f"Loading ERA5 reanalysis anchors from {PATH_ERA5}...")
        era5_raw = pd.read_csv(PATH_ERA5)
        time_col = 'valid_time' if 'valid_time' in era5_raw.columns else 'datetime'
        if time_col in era5_raw.columns:
            era5_raw['datetime'] = pd.to_datetime(era5_raw[time_col])
            era5_df = era5_raw.set_index('datetime').resample('1min').interpolate(method='time')
            logger.info(f"Interpolated ERA5 anchors: {len(era5_df)} 1-minute rows.")

    # 5. Fit Scalers Across Node Corpus
    corpus_combined = pd.concat(list(node_data.values()), axis=0)
    preprocessor.fit_scalers(corpus_combined)
    joblib.dump(preprocessor.scalers, os.path.join(OUTPUTS_DIR, 'model', 'scalers.joblib'))
    joblib.dump(node_name_map, os.path.join(OUTPUTS_DIR, 'model', 'node_map.joblib'))
    logger.info("Saved scalers and node mapping to outputs/model/")

    # 6. Generate Multi-Input Sequences
    all_X_cont = []
    all_node_ids = []
    all_Y_targets = []

    for node_id, df_node in node_data.items():
        node_int = node_name_map[node_id]
        
        # Subsample for quick local test mode if requested
        if args.mode == 'LOCAL_TEST':
            df_node = df_node.tail(15000)
            
        # Add cyclical time
        df_feat = preprocessor.add_cyclical_features(df_node)
        df_scaled = preprocessor.transform(df_feat)
        
        # Build features table
        feat_cols = target_vars + ['sin_hour', 'cos_hour', 'sin_day', 'cos_day']
        
        # Add ERA5 anchor columns if available
        if era5_df is not None:
            aligned_era5 = era5_df.reindex(df_scaled.index).interpolate().bfill().ffill()
            for ecol in ['temperature_2m', 'relative_humidity_2m', 'surface_pressure']:
                if ecol in aligned_era5.columns:
                    df_scaled[f'era5_{ecol}'] = aligned_era5[ecol]
                    feat_cols.append(f'era5_{ecol}')
                    
        # Replace NaNs with 0 for model feeding
        df_features = df_scaled[feat_cols].copy()
        df_features = df_features.fillna(0.0)
        
        # Generate sliding windows
        step = 5 if args.mode == 'LOCAL_TEST' else 2
        X_seq, node_seq = preprocessor.generate_sliding_windows(df_features, node_int, step=step)
        
        # Target is the 4 meteorological variables
        Y_target_seq = X_seq[..., :len(target_vars)]
        
        if len(X_seq) > 0:
            all_X_cont.append(X_seq)
            all_node_ids.append(node_seq)
            all_Y_targets.append(Y_target_seq)

    if not all_X_cont:
        logger.error("No valid sequences generated.")
        sys.exit(1)

    X_continuous = np.concatenate(all_X_cont, axis=0)
    X_node = np.concatenate(all_node_ids, axis=0)
    Y_targets = np.concatenate(all_Y_targets, axis=0)
    
    logger.info(f"Dataset generated: Total sequences = {len(X_continuous):,}, Shape = {X_continuous.shape[1:]}")

    # Train/Validation Split (85% Train, 15% Val)
    n_samples = len(X_continuous)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.85 * n_samples)
    
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_cont, X_val_cont = X_continuous[train_idx], X_continuous[val_idx]
    X_train_node, X_val_node = X_node[train_idx], X_node[val_idx]
    Y_train, Y_val = Y_targets[train_idx], Y_targets[val_idx]

    # 7. Construct and Compile Universal Bi-LSTM
    num_features = X_continuous.shape[-1]
    model = build_universal_bilstm_model(
        lookback=args.lookback,
        num_continuous_features=num_features,
        num_nodes=max(8, len(node_name_map)),
        embedding_dim=4,
        num_targets=len(target_vars),
        lstm_1=64,
        lstm_2=32,
        dropout_rate=0.2,
        learning_rate=1e-3
    )
    model.summary(print_fn=logger.info)

    # 8. Training Callbacks
    model_save_path = os.path.join(OUTPUTS_DIR, 'model', 'universal_bilstm.keras')
    cbs = [
        tf_callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        tf_callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    logger.info(f"Starting Training: Epochs={epochs}, Batch Size={batch_size}...")
    history = model.fit(
        x=[X_train_cont, X_train_node],
        y=Y_train,
        validation_data=([X_val_cont, X_val_node], Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1
    )

    # 9. Save History & Loss Plot
    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(os.path.join(OUTPUTS_DIR, 'metrics', 'training_history.csv'), index=False)
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(df_hist['loss'], label='Train Loss (Mask-Weighted MSE)', color='#2563eb', linewidth=2)
    plt.plot(df_hist['val_loss'], label='Validation Loss', color='#ef4444', linewidth=2)
    plt.title('Universal Bi-LSTM Imputation Loss Curve (Nodes 3 & 5)', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'loss_curve.png'), bbox_inches='tight')
    plt.close()

    logger.info(f"[SUCCESS] Model training complete. Artifacts saved in {OUTPUTS_DIR}")


if __name__ == '__main__':
    main()
