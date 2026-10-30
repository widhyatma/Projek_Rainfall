"""
Spatial Cross-Node Transfer Engine
===================================
Provides high-accuracy spatial transfer and regression calibration between
neighboring AWS micro-stations in the network.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class SpatialTransferEngine:
    """
    Engine for reconstructing target station gaps using active neighboring companion nodes.
    """
    def __init__(self, target_vars=None):
        self.target_vars = target_vars or ['temperature', 'humidity', 'pressure', 'dewpoint']
        self.models = {}

    def fit_transfer_model(self, df_target, df_companion, var_name):
        """
        Fits a Ridge regression model calibrated with diurnal harmonic features.
        """
        # Align overlapping valid minutes
        combined = pd.DataFrame({
            'target': df_target[var_name],
            'companion': df_companion[var_name]
        }).dropna()
        
        if len(combined) < 60:
            raise ValueError(f"Insufficient overlapping valid minutes ({len(combined)}) to calibrate spatial model.")
            
        # Feature Engineering: Companion observation + Diurnal harmonics
        hour = combined.index.hour + combined.index.minute / 60.0
        X = np.column_stack([
            combined['companion'].values,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0)
        ])
        y = combined['target'].values
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        self.models[var_name] = model
        return model

    def predict(self, df_companion, var_name):
        """
        Predicts target station values from companion station stream.
        """
        if var_name not in self.models:
            raise KeyError(f"Spatial model for '{var_name}' has not been fitted yet.")
            
        model = self.models[var_name]
        hour = df_companion.index.hour + df_companion.index.minute / 60.0
        X = np.column_stack([
            df_companion[var_name].values,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0)
        ])
        return model.predict(X)

    @staticmethod
    def calculate_correlation_matrix(node_dict, var_name='temperature'):
        """
        Calculates Pearson correlation coefficient between all stations in the network.
        """
        df_corr = pd.DataFrame({node_id: df[var_name] for node_id, df in node_dict.items() if var_name in df.columns})
        return df_corr.corr()
