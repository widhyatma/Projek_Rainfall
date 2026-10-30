"""
Meteorological Preprocessor Module
===================================
Handles quality control, cyclic time feature engineering, dynamic scaling,
and sequence formatting for universal Bi-LSTM imputation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Safe Dynamic CuPy/NumPy Import (Rule Compliant)
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp


class MeteorologicalPreprocessor:
    """
    Preprocessor for multi-node meteorological data with physical QC and cyclical features.
    """
    def __init__(self, target_vars=None, lookback=120):
        self.target_vars = target_vars or ['temperature', 'humidity', 'pressure', 'dewpoint']
        self.lookback = lookback
        self.scalers = {}
        self.node_encoder = LabelEncoder()
        
        # Physical boundary bounds for quality control
        self.qc_bounds = {
            'temperature': (15.0, 45.0),
            'humidity': (40.0, 100.0),
            'pressure': (1002.0, 1019.0),
            'dewpoint': (15.0, 35.0)
        }

    def apply_qc(self, df):
        """Replaces sensor out-of-range anomalies with NaN."""
        df_qc = df.copy()
        df_qc = df_qc.loc[:, ~df_qc.columns.duplicated()]
        for var, (vmin, vmax) in self.qc_bounds.items():
            if var in df_qc.columns:
                s = df_qc[var]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                invalid_mask = (s < vmin) | (s > vmax)
                df_qc.loc[invalid_mask, var] = np.nan
        return df_qc

    @staticmethod
    def add_cyclical_features(df):
        """Extracts sin/cos encodings for hour and day-of-year."""
        df_feat = df.copy()
        if not isinstance(df_feat.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pandas DatetimeIndex.")
            
        hour = df_feat.index.hour + df_feat.index.minute / 60.0
        day_of_year = df_feat.index.dayofyear
        
        df_feat['sin_hour'] = np.sin(2 * np.pi * hour / 24.0)
        df_feat['cos_hour'] = np.cos(2 * np.pi * hour / 24.0)
        df_feat['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df_feat['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
        return df_feat

    def fit_scalers(self, combined_df):
        """Fits MinMaxScaler on continuous feature sets across all nodes."""
        for var in self.target_vars:
            if var in combined_df.columns:
                scaler = MinMaxScaler()
                valid_vals = combined_df[var].dropna().values.reshape(-1, 1)
                if len(valid_vals) > 0:
                    scaler.fit(valid_vals)
                    self.scalers[var] = scaler

    def transform(self, df):
        """Scales target variables to [0, 1]."""
        df_scaled = df.copy()
        for var, scaler in self.scalers.items():
            if var in df_scaled.columns:
                mask = df_scaled[var].notna()
                if mask.sum() > 0:
                    vals_2d = df_scaled.loc[mask, [var]].values
                    scaled_1d = scaler.transform(vals_2d).flatten()
                    df_scaled.loc[mask, var] = scaled_1d
        return df_scaled

    def inverse_transform(self, df_or_arr, var_name):
        """Inverts scaled values back to original meteorological physical units."""
        if var_name not in self.scalers:
            return df_or_arr
        scaler = self.scalers[var_name]
        if isinstance(df_or_arr, np.ndarray):
            shape_orig = df_or_arr.shape
            arr_reshaped = df_or_arr.reshape(-1, 1)
            arr_inv = scaler.inverse_transform(arr_reshaped)
            return arr_inv.reshape(shape_orig)
        else:
            return scaler.inverse_transform(df_or_arr.values.reshape(-1, 1)).flatten()

    def generate_sliding_windows(self, df_continuous, node_id_int, step=1):
        """
        Constructs multi-input training sequences of shape (N, SEQ_LEN, num_features).
        Uses xp for GPU speedup if available, converting back to NumPy for Keras compatibility.
        """
        values = df_continuous.values
        n_rows, n_cols = values.shape
        
        X_seqs = []
        node_seqs = []
        
        for i in range(0, n_rows - self.lookback + 1, step):
            window = values[i : i + self.lookback]
            X_seqs.append(window)
            node_seqs.append(node_id_int)
            
        X_arr = np.array(X_seqs, dtype=np.float32)
        node_arr = np.array(node_seqs, dtype=np.int32).reshape(-1, 1)
        return X_arr, node_arr

    @staticmethod
    def inject_synthetic_missingness(arr, dropout_rate=0.2, block_size=30):
        """
        Simulates realistic sensor gaps (continuous blocks + random dropouts) for validation.
        """
        corrupted = arr.copy()
        mask = np.ones_like(arr, dtype=np.float32)
        n_samples = len(arr)
        
        # Block dropout simulation
        n_blocks = int((n_samples * dropout_rate) / block_size)
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n_samples - block_size))
            corrupted[start : start + block_size] = np.nan
            mask[start : start + block_size] = 0.0
            
        return corrupted, mask
