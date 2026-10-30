"""
Physics-Informed Post-Processing & Smoothing Module
===================================================
Enforces thermodynamic atmospheric laws and applies zero-phase Savitzky-Golay filtering.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class PhysicsPostProcessor:
    """
    Sanitizes imputed meteorological time series with thermodynamic constraints
    and eliminates high-frequency sensor quantization jitter.
    """
    def __init__(self, window_length=15, polyorder=2):
        self.window_length = window_length if window_length % 2 == 1 else window_length + 1
        self.polyorder = polyorder

    def apply_smoothing(self, df_in):
        """
        Applies Savitzky-Golay filter across meteorological parameters.
        """
        df = df_in.copy()
        
        for col in ['temperature', 'humidity', 'pressure', 'dewpoint', 'dew']:
            if col in df.columns and len(df[col].dropna()) >= self.window_length:
                # Interpolate internal NaNs briefly for smooth filtering
                s = df[col].interpolate(method='linear').ffill().bfill()
                df[col] = savgol_filter(s, window_length=self.window_length, polyorder=self.polyorder)
                
        return df

    def enforce_thermodynamics(self, df_in):
        """
        Enforces physical laws of atmospheric thermodynamics:
        1. Magnus-Tetens: T_dew <= T_ambient
        2. Relative Humidity: 10% <= RH <= 100%
        3. Atmospheric Surface Pressure: 900 hPa <= P <= 1050 hPa
        """
        df = df_in.copy()
        
        # 1. Relative Humidity bounding
        if 'humidity' in df.columns:
            df['humidity'] = df['humidity'].clip(lower=10.0, upper=100.0)
            
        # 2. Surface Pressure bounding
        if 'pressure' in df.columns:
            df['pressure'] = df['pressure'].clip(lower=900.0, upper=1050.0)
            
        # 3. Magnus-Tetens Dew Point constraint
        dew_col = 'dewpoint' if 'dewpoint' in df.columns else ('dew' if 'dew' in df.columns else None)
        if dew_col and 'temperature' in df.columns:
            df[dew_col] = np.minimum(df[dew_col], df['temperature'])
            
        return df

    def process(self, df_in):
        """Executes full physics postprocessing pipeline."""
        df_smooth = self.apply_smoothing(df_in)
        df_phys = self.enforce_thermodynamics(df_smooth)
        return df_phys
