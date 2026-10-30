#!/usr/bin/env python
"""
Universal Imputation Pipeline (CLI & Python API)
================================================
Automated multi-phase gap detection, Bi-LSTM reconstruction, and physics-informed post-processing.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Add framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framework.preprocessor import MeteorologicalPreprocessor
from framework.spatial_transfer import SpatialTransferEngine
from framework.physics_postprocess import PhysicsPostProcessor
from framework.network import mask_weighted_mse

import tensorflow as tf


class UniversalImputationPipeline:
    """
    End-to-End Imputation Engine combining Splines, Bi-LSTM, Spatial Transfer, and Physics Post-Processing.
    """
    def __init__(self, model_dir=None, lookback=120):
        self.lookback = lookback
        self.target_vars = ['temperature', 'humidity', 'pressure', 'dewpoint']
        self.preprocessor = MeteorologicalPreprocessor(target_vars=self.target_vars, lookback=lookback)
        self.physics_postprocessor = PhysicsPostProcessor(window_length=15, polyorder=2)
        self.spatial_engine = SpatialTransferEngine(target_vars=self.target_vars)
        
        # Load weights and scalers if available
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(base_dir, "outputs", "model")
        
        self.model = None
        self.scalers = None
        self.node_map = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.model_dir, "universal_bilstm.keras")
        scalers_path = os.path.join(self.model_dir, "scalers.joblib")
        node_map_path = os.path.join(self.model_dir, "node_map.joblib")
        
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path, custom_objects={'mask_weighted_mse': mask_weighted_mse})
                print("[INFO] Successfully loaded Universal Bi-LSTM model.")
            except Exception as e:
                print(f"[WARN] Failed to load Keras model: {e}")
                
        if os.path.exists(scalers_path):
            self.scalers = joblib.load(scalers_path)
            self.preprocessor.scalers = self.scalers
            
        if os.path.exists(node_map_path):
            self.node_map = joblib.load(node_map_path)

    def impute_dataframe(self, df_target, node_id='id-03', df_companion=None, df_era5=None):
        """
        Executes full 2-phase hybrid imputation on an input DataFrame.
        """
        df = df_target.copy()
        
        # Standardize column names
        col_map = {'dew': 'dewpoint', 'temp': 'temperature', 'hum': 'humidity', 'press': 'pressure'}
        df = df.rename(columns=col_map)
        
        # Ensure DatetimeIndex at 1-minute continuous grid
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
                df = df.set_index('datetime')
            elif 'datetime_wib' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime_wib'])
                df = df.set_index('datetime')
            else:
                raise ValueError("DataFrame must contain 'timestamp', 'datetime_wib', or have a DatetimeIndex.")
                
        # Resample to 1-minute mean before reindexing full continuous grid
        df_1min = df[self.target_vars].resample('1min').mean()
        full_grid = pd.date_range(start=df_1min.index.min(), end=df_1min.index.max(), freq='1min')
        df_grid = df_1min.reindex(full_grid)
        
        # Identify Gaps
        is_missing = df_grid['temperature'].isna()
        
        # Ensure Companion also has DatetimeIndex and 1-min resampled grid
        if df_companion is not None and not df_companion.empty:
            df_comp_aligned = df_companion.copy()
            df_comp_aligned = df_comp_aligned.rename(columns=col_map)
            if not isinstance(df_comp_aligned.index, pd.DatetimeIndex):
                if 'timestamp' in df_comp_aligned.columns:
                    df_comp_aligned['datetime'] = pd.to_datetime(df_comp_aligned['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
                    df_comp_aligned = df_comp_aligned.set_index('datetime')
                elif 'datetime_wib' in df_comp_aligned.columns:
                    df_comp_aligned['datetime'] = pd.to_datetime(df_comp_aligned['datetime_wib'])
                    df_comp_aligned = df_comp_aligned.set_index('datetime')
            comp_vars = [c for c in self.target_vars if c in df_comp_aligned.columns]
            df_companion = df_comp_aligned[comp_vars].resample('1min').mean()

        # 1. PHASE 1: Fast Spline Imputation for Micro-Gaps (<= 15 minutes)
        df_imputed = df_grid.copy()
        df_imputed['is_imputed'] = is_missing.astype(int)
        
        # Linear/Spline interpolation for micro gaps
        df_imputed[self.target_vars] = df_imputed[self.target_vars].interpolate(method='time', limit=15)
        
        # 2. PHASE 2: Macro-Gaps (> 15 minutes)
        remaining_missing = df_imputed['temperature'].isna()
        if remaining_missing.sum() > 0:
            print(f"[INFO] Processing {remaining_missing.sum()} macro-gap minutes...")
            
            # Strategy A: Companion Spatial Transfer (Highest Fidelity if Companion Available)
            if df_companion is not None and not df_companion.empty:
                print("[INFO] Applying Phase-2 Spatial Transfer from companion node...")
                for var in self.target_vars:
                    if var in df_companion.columns:
                        try:
                            # Align valid history to fit transfer model
                            self.spatial_engine.fit_transfer_model(df_grid, df_companion, var)
                            pred_vals = self.spatial_engine.predict(df_companion.reindex(full_grid), var)
                            df_imputed.loc[remaining_missing, var] = pred_vals[remaining_missing]
                        except Exception as e:
                            print(f"[WARN] Spatial transfer fallback failed for {var}: {e}")
                            
            # Strategy B: Deep Bi-LSTM Autoencoder (If No Active Companion or Standalone Outage)
            elif self.model is not None and self.scalers is not None:
                print("[INFO] Applying Phase-2 Universal Bi-LSTM Autoencoder...")
                # Extract features
                df_feat = self.preprocessor.add_cyclical_features(df_imputed)
                df_scaled = self.preprocessor.transform(df_feat)
                
                # Fill missing inputs with 0 for autoencoding
                df_in = df_scaled.fillna(0.0)
                node_int = self.node_map.get(node_id, 0) if self.node_map else 0
                
                # Sliding window inference
                X_seq, node_seq = self.preprocessor.generate_sliding_windows(df_in[self.target_vars + ['sin_hour', 'cos_hour', 'sin_day', 'cos_day']], node_int)
                if len(X_seq) > 0:
                    y_pred_seq = self.model.predict([X_seq, node_seq], verbose=0)
                    # Inverse scale and map back
                    for idx, var in enumerate(self.target_vars):
                        pred_var = y_pred_seq[..., idx].mean(axis=1)
                        pred_phys = self.preprocessor.inverse_transform(pred_var, var)
                        df_imputed.loc[df_imputed[var].isna(), var] = pred_phys[:df_imputed[var].isna().sum()]
                        
            # Strategy C: Long Polynomial Spline Fallback
            df_imputed[self.target_vars] = df_imputed[self.target_vars].interpolate(method='polynomial', order=2).bfill().ffill()

        # 3. PHASE 3: Physics-Informed Post-Processing & Savitzky-Golay Smoothing
        print("[INFO] Applying Physics-Informed Post-Processing & Smoothing...")
        df_final = self.physics_postprocessor.process(df_imputed)
        
        # Round values to standard meteorological precision
        for var in self.target_vars:
            if var in df_final.columns:
                df_final[var] = df_final[var].round(2)
                
        # Re-add timestamp and datetime_wib columns
        df_final['datetime_wib'] = df_final.index.strftime('%Y-%m-%d %H:%M:%S')
        df_final['timestamp'] = ((pd.to_datetime(df_final.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') - 7 * 3600).astype(int)
        
        return df_final


def main():
    parser = argparse.ArgumentParser(description="Run Universal Imputation Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Input CSV path with missing data")
    parser.add_argument('--node', type=str, default='id-03', help="Station Node ID (e.g. id-01, id-03, id-05)")
    parser.add_argument('--companion', type=str, default=None, help="Companion CSV path (optional)")
    parser.add_argument('--output', type=str, required=True, help="Output clean CSV path")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    df_comp = pd.read_csv(args.companion) if args.companion and os.path.exists(args.companion) else None
    
    pipeline = UniversalImputationPipeline()
    df_clean = pipeline.impute_dataframe(df_in, node_id=args.node, df_companion=df_comp)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    print(f"[SUCCESS] Clean reconstructed dataset saved to {args.output}")


if __name__ == '__main__':
    main()
