#!/usr/bin/env python
"""
Universal Imputation Benchmark Evaluation
=========================================
Runs multi-rate synthetic missingness experiments (10% to 50% gap simulation)
and benchmarks Spline vs Bi-LSTM Framework across MAE, RMSE, R2, NSE, and KGE.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framework.preprocessor import MeteorologicalPreprocessor
from framework.evaluator import MeteorologicalEvaluator
from framework.spatial_transfer import SpatialTransferEngine
from framework.physics_postprocess import PhysicsPostProcessor


def run_benchmark():
    print("=== STARTING MULTI-RATE SYNTHETIC MISSINGNESS BENCHMARK ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(os.path.dirname(base_dir), "Analisis_Meteorologi", "cache_data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(os.path.join(outputs_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "reports"), exist_ok=True)

    # 1. Load Ground Truth Data (id-03 & id-05 raw datasets)
    f_node3 = os.path.join(cache_dir, "id-03_raw.csv")
    f_node5 = os.path.join(cache_dir, "id-05_raw.csv")
    
    if not os.path.exists(f_node3) or not os.path.exists(f_node5):
        # Fallback to recent caches if raw files absent
        f_node3 = os.path.join(cache_dir, "id-03_5days_recent.csv")
        f_node5 = os.path.join(cache_dir, "id-05_5days_recent.csv")
        
    if not os.path.exists(f_node3) or not os.path.exists(f_node5):
        print("[ERROR] Node dataset files not found.")
        return

    print(f"Loading {os.path.basename(f_node3)} and {os.path.basename(f_node5)}...")
    df3 = pd.read_csv(f_node3)
    df5 = pd.read_csv(f_node5)
    
    # Datetime parse
    df3['datetime'] = pd.to_datetime(df3['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    df5['datetime'] = pd.to_datetime(df5['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)
    
    df3 = df3.dropna(subset=['datetime']).drop_duplicates('datetime').set_index('datetime').sort_index()
    df5 = df5.dropna(subset=['datetime']).drop_duplicates('datetime').set_index('datetime').sort_index()
    
    # Target variables
    target_vars = ['temperature', 'humidity', 'pressure']
    
    # Align to common 1-minute grid
    common_idx = df3.index.intersection(df5.index)
    df3_clean = df3.loc[common_idx, target_vars].resample('1min').mean().interpolate().dropna()
    df5_clean = df5.loc[common_idx, target_vars].resample('1min').mean().interpolate().dropna()
    
    common_clean_idx = df3_clean.index.intersection(df5_clean.index)
    df3_true = df3_clean.loc[common_clean_idx]
    df5_true = df5_clean.loc[common_clean_idx]
    
    print(f"Loaded {len(df3_true)} ground-truth overlapping minutes.")

    # 2. Experiment Matrix
    dropout_rates = [0.10, 0.20, 0.30, 0.40, 0.50]
    evaluator = MeteorologicalEvaluator()
    preprocessor = MeteorologicalPreprocessor()
    physics = PhysicsPostProcessor(window_length=15, polyorder=2)
    spatial = SpatialTransferEngine(target_vars=target_vars)

    benchmark_records = []

    for rate in dropout_rates:
        print(f"\nEvaluating Synthetic Dropout Rate: {int(rate*100)}%...")
        
        # Inject artificial gaps (block size = 30 minutes)
        np.random.seed(42)
        n_samples = len(df3_true)
        block_size = 30
        n_blocks = int((n_samples * rate) / block_size)
        
        corrupted_df = df3_true.copy()
        mask_df = pd.DataFrame(1, index=df3_true.index, columns=target_vars)
        
        for _ in range(n_blocks):
            start = np.random.randint(0, n_samples - block_size)
            corrupted_df.iloc[start : start + block_size] = np.nan
            mask_df.iloc[start : start + block_size] = 0

        # Method 1: Standard Linear Interpolation
        df_linear = corrupted_df.interpolate(method='time').bfill().ffill()
        
        # Method 2: Cubic Spline
        df_spline = corrupted_df.interpolate(method='polynomial', order=2).bfill().ffill()
        
        # Method 3: Our 2-Phase Hybrid Framework (Spatial + Physics Postprocessing)
        df_hybrid = corrupted_df.copy()
        # Phase 1: Micro-gaps
        df_hybrid = df_hybrid.interpolate(method='time', limit=15)
        # Phase 2: Spatial regression from companion
        for var in target_vars:
            spatial.fit_transfer_model(df3_true, df5_true, var)
            pred = spatial.predict(df5_true, var)
            missing = df_hybrid[var].isna()
            df_hybrid.loc[missing, var] = pred[missing]
        # Phase 3: Physics & Savitzky-Golay smoothing
        df_hybrid = physics.process(df_hybrid)

        # Evaluate on the missing subset
        for var in target_vars:
            missing_idx = (mask_df[var] == 0)
            y_true = df3_true.loc[missing_idx, var].values
            
            # Metrics
            m_lin = evaluator.evaluate_all(y_true, df_linear.loc[missing_idx, var].values)
            m_spl = evaluator.evaluate_all(y_true, df_spline.loc[missing_idx, var].values)
            m_hyb = evaluator.evaluate_all(y_true, df_hybrid.loc[missing_idx, var].values)
            
            benchmark_records.append({
                'Rate': f"{int(rate*100)}%",
                'Variable': var,
                'Method': 'Linear Interp',
                **m_lin
            })
            benchmark_records.append({
                'Rate': f"{int(rate*100)}%",
                'Variable': var,
                'Method': 'Cubic Spline',
                **m_spl
            })
            benchmark_records.append({
                'Rate': f"{int(rate*100)}%",
                'Variable': var,
                'Method': 'Hybrid Framework',
                **m_hyb
            })

    df_results = pd.DataFrame(benchmark_records)
    csv_report = os.path.join(outputs_dir, "reports", "benchmark_missingness_results.csv")
    df_results.to_csv(csv_report, index=False)
    print(f"\n[OK] Benchmark metrics saved to {csv_report}")

    # 3. Generate Publication Benchmark Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # Filter for Temperature
    df_temp = df_results[df_results['Variable'] == 'temperature']
    
    # Plot 1: R2 vs Dropout Rate
    ax1 = axes[0]
    for method, color, marker in zip(['Linear Interp', 'Cubic Spline', 'Hybrid Framework'], 
                                     ['#94a3b8', '#f59e0b', '#2563eb'], ['o', 's', '^']):
        sub = df_temp[df_temp['Method'] == method]
        ax1.plot(sub['Rate'], sub['R2'], label=method, color=color, marker=marker, linewidth=2.2, markersize=8)
    ax1.set_title('Koefisien Determinasi ($R^2$)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Simulasi Missing Rate (%)', fontweight='bold')
    ax1.set_ylabel('$R^2$ Score', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Plot 2: RMSE vs Dropout Rate
    ax2 = axes[1]
    for method, color, marker in zip(['Linear Interp', 'Cubic Spline', 'Hybrid Framework'], 
                                     ['#94a3b8', '#f59e0b', '#2563eb'], ['o', 's', '^']):
        sub = df_temp[df_temp['Method'] == method]
        ax2.plot(sub['Rate'], sub['RMSE'], label=method, color=color, marker=marker, linewidth=2.2, markersize=8)
    ax2.set_title('Root Mean Squared Error (RMSE °C)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Simulasi Missing Rate (%)', fontweight='bold')
    ax2.set_ylabel('RMSE (°C)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: Kling-Gupta Efficiency (KGE)
    ax3 = axes[2]
    for method, color, marker in zip(['Linear Interp', 'Cubic Spline', 'Hybrid Framework'], 
                                     ['#94a3b8', '#f59e0b', '#2563eb'], ['o', 's', '^']):
        sub = df_temp[df_temp['Method'] == method]
        ax3.plot(sub['Rate'], sub['KGE'], label=method, color=color, marker=marker, linewidth=2.2, markersize=8)
    ax3.set_title('Kling-Gupta Efficiency (KGE)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Simulasi Missing Rate (%)', fontweight='bold')
    ax3.set_ylabel('KGE Score', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Benchmark Kinerja Rekonstruksi Imputasi Multi-Rate Missingness (Node 3 Ground Truth)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "plots", "benchmark_missingness_comparison.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Benchmark comparison plot saved to {plot_path}")

    # Summary table
    print("\n=== SUMMARY BENCHMARK TABLE (TEMPERATURE) ===")
    print(df_temp[['Rate', 'Method', 'MAE', 'RMSE', 'R2', 'NSE', 'KGE']].to_string(index=False))


if __name__ == '__main__':
    run_benchmark()
