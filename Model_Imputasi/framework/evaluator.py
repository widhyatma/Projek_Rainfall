"""
Meteorological Evaluation Module
=================================
Calculates hydrometeorological benchmark metrics: MAE, RMSE, R2, NSE, and KGE.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MeteorologicalEvaluator:
    """
    Evaluates imputation accuracy across international hydrometeorological standards.
    """
    @staticmethod
    def nse(y_true, y_pred):
        """Nash-Sutcliffe Efficiency (NSE)."""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            return np.nan
        denominator = np.sum((yt - np.mean(yt)) ** 2)
        if denominator == 0:
            return 1.0 if np.allclose(yt, yp) else -np.inf
        return 1.0 - (np.sum((yt - yp) ** 2) / denominator)

    @staticmethod
    def kge(y_true, y_pred):
        """Kling-Gupta Efficiency (KGE)."""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) < 2:
            return np.nan
            
        r = np.corrcoef(yt, yp)[0, 1]
        alpha = np.std(yp) / (np.std(yt) + 1e-7)
        beta = np.mean(yp) / (np.mean(yt) + 1e-7)
        
        kge_val = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
        return kge_val

    def evaluate_all(self, y_true, y_pred):
        """Calculates full metric suite for a single variable."""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        
        if len(yt) == 0:
            return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        nse_val = self.nse(yt, yp)
        kge_val = self.kge(yt, yp)
        
        return {
            'MAE': round(float(mae), 3),
            'RMSE': round(float(rmse), 3),
            'R2': round(float(r2), 4),
            'NSE': round(float(nse_val), 4),
            'KGE': round(float(kge_val), 4)
        }

    def evaluate_dataframe(self, df_true, df_pred, var_cols):
        """Evaluates multiple meteorological variables and returns a formatted DataFrame."""
        results = {}
        for col in var_cols:
            if col in df_true.columns and col in df_pred.columns:
                results[col] = self.evaluate_all(df_true[col].values, df_pred[col].values)
        return pd.DataFrame(results).T
