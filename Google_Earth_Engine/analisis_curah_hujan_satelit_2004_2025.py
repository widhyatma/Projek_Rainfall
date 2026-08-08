#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🛰️ Analisis Komparasi Menyeluruh Curah Hujan Satelit & Reanalisis Harian (2004 – 2025)
Matriks Komparasi Antar-Variabel (All-to-All Pairwise Inter-Comparison) & Benchmark CHIRPS Reanalisis.
Membandingkan CHIRPS_RNL, CHIRPS_SAT, GSMaP, IMERG, PERSIANN, ERA5, ERA5_LAND, dan Observasi Pos Hujan OYA.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style='whitegrid')

# ─── 1. Inisialisasi Path & Direktori Output ───
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.path.abspath('Google_Earth_Engine')
data_csv = os.path.join(base_dir, 'Data_Satelit', 'Data_Curah_Hujan_Satelit_2004_2025.csv')
out_dir = os.path.join(base_dir, 'Hasil_Analisis_Harian_2004_2025')
os.makedirs(out_dir, exist_ok=True)

print(f"Base Directory : {base_dir}")
print(f"Data File      : {data_csv}")
print(f"Output Folder  : {out_dir}")

# ─── 2. Memuat Data & Standarisasi Waktu ───
if not os.path.exists(data_csv):
    raise FileNotFoundError(f"File data tidak ditemukan di: {data_csv}")

df_raw = pd.read_csv(data_csv)
print(f"Total baris data mentah: {len(df_raw):,}")

# Parsing datetime
if 'datetime_utc' in df_raw.columns:
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime_utc'], utc=True)
elif 'unixtime' in df_raw.columns:
    df_raw['datetime'] = pd.to_datetime(df_raw['unixtime'], unit='ms' if df_raw['unixtime'].max() > 1e11 else 's', utc=True)

df = df_raw.sort_values('datetime').set_index('datetime')

# Buat versi ERA5_LAND terkoreksi skala untuk visualisasi komparatif yang adil
if 'ERA5_LAND' in df.columns and df['ERA5_LAND'].mean() > 30:
    df['ERA5_LAND_SCALED'] = df['ERA5_LAND'] / 14.498

sat_cols = ['CHIRPS_RNL', 'CHIRPS_SAT', 'GSMaP', 'IMERG', 'PERSIANN', 'ERA5', 'ERA5_LAND', 'OYA']
avail_cols = [c for c in sat_cols if c in df.columns]
df_eval = df[avail_cols].copy()

print("Kolom yang dianalisis:", avail_cols)
print(f"Rentang Waktu: {df_eval.index.min()} s/d {df_eval.index.max()} ({len(df_eval):,} hari)")

# ─── 3. Statistik Deskriptif & Audit Data ───
desc_stats = df_eval.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc_stats['skewness'] = df_eval.skew()
desc_stats['kurtosis'] = df_eval.kurtosis()
desc_stats['null_count'] = df_eval.isna().sum()
desc_stats['null_pct'] = (df_eval.isna().sum() / len(df_eval)) * 100.0
desc_stats['dry_days_pct'] = ((df_eval < 0.1).sum() / len(df_eval)) * 100.0

desc_stats.to_csv(os.path.join(out_dir, 'statistik_deskriptif_harian.csv'))
print("\n=== STATISTIK DESKRIPTIF DATA HARIAN (mm/hari) ===")
print(desc_stats.round(3).to_string())

# ─── 4. Matriks Scatter Plot Pairwise All-to-All (8x8) ───
print("\nMembuat Scatter Plot Matrix Pairwise (All-to-All)...")
n_cols = len(avail_cols)
fig_size = 2.8 * n_cols
fig, axes = plt.subplots(n_cols, n_cols, figsize=(fig_size, fig_size), squeeze=False)
fig.suptitle('Scatter Plot Matrix — Korelasi Curah Hujan Harian All-to-All (2004 – 2025)\nSemua Kombinasi Pasangan Satelit, Reanalisis (ERA5, ERA5-Land), dan Pos Hujan Oya',
             fontsize=16, fontweight='bold', y=1.01)

COLORS = plt.cm.tab10.colors

for i, col_y in enumerate(avail_cols):
    for j, col_x in enumerate(avail_cols):
        ax = axes[i][j]
        if i == j:
            data_diag = df_eval[col_y].dropna()
            ax.hist(data_diag, bins=35, color=COLORS[i % len(COLORS)], alpha=0.75, edgecolor='white', linewidth=0.5)
            ax.set_facecolor('#f7f9fa')
            ax.set_title(col_y, fontsize=9.5, fontweight='bold', pad=3)
            ax.set_xlabel('mm/hari', fontsize=7)
            ax.set_ylabel('Freq', fontsize=7)
            ax.tick_params(labelsize=6.5)
        else:
            mask = df_eval[col_x].notna() & df_eval[col_y].notna()
            x_data = df_eval.loc[mask, col_x]
            y_data = df_eval.loc[mask, col_y]
            
            if len(x_data) < 5:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                continue
                
            r_p, _ = stats.pearsonr(x_data, y_data)
            r_s, _ = stats.spearmanr(x_data, y_data)
            
            n_plot = min(len(x_data), 3000)
            idx_samp = np.random.choice(len(x_data), n_plot, replace=False) if len(x_data) > n_plot else np.arange(len(x_data))
            ax.scatter(x_data.iloc[idx_samp], y_data.iloc[idx_samp], alpha=0.20, s=6, color=COLORS[j % len(COLORS)], edgecolors='none')
            
            # Garis 1:1
            joint_min = 0
            joint_max = max(x_data.max(), y_data.max())
            ax.plot([joint_min, joint_max], [joint_min, joint_max], 'k--', linewidth=0.9, alpha=0.6)
            
            # Garis OLS
            slope, intercept, _, _, _ = stats.linregress(x_data, y_data)
            x_line = np.linspace(joint_min, joint_max, 50)
            ax.plot(x_line, slope * x_line + intercept, color='crimson', linewidth=1.1)
            
            txt = f"r={r_p:.2f}\nρ={r_s:.2f}"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=6.8, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            ax.tick_params(labelsize=6.5)
            
        if i == n_cols - 1:
            ax.set_xlabel(avail_cols[j], fontsize=8.5, fontweight='bold', labelpad=3)
        else:
            ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel(avail_cols[i], fontsize=8.5, fontweight='bold', labelpad=3)
        else:
            ax.set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '01_scatterplot_matrix_curah_hujan.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 5. Matriks Heatmap Korelasi Pearson ($r$) & Spearman ($\rho$) ───
print("Membuat Heatmap Korelasi Pearson & Spearman...")
corr_pearson = df_eval[avail_cols].corr(method='pearson').round(3)
corr_spearman = df_eval[avail_cols].corr(method='spearman').round(3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, square=True,
            linewidths=0.8, linecolor='white', annot_kws={'size': 10, 'weight': 'bold'}, ax=ax1,
            cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8})
ax1.set_title('Matriks Korelasi Pearson (r)\nLinear Relationship (All-to-All)', fontsize=13, fontweight='bold', pad=12)
ax1.tick_params(axis='x', rotation=35)

sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1, square=True,
            linewidths=0.8, linecolor='white', annot_kws={'size': 10, 'weight': 'bold'}, ax=ax2,
            cbar_kws={'label': 'Spearman Correlation (ρ)', 'shrink': 0.8})
ax2.set_title('Matriks Korelasi Spearman (ρ)\nRank-Based Monotonic Relationship (All-to-All)', fontsize=13, fontweight='bold', pad=12)
ax2.tick_params(axis='x', rotation=35)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '02_heatmap_korelasi_pearson_spearman.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 6. Matriks Lengkap Evaluasi Metrik All-to-All ($8 \times 8$) ───
print("Menghitung Matriks Evaluasi Lengkap Inter-Model All-to-All...")
pairwise_all = []
rmse_matrix = pd.DataFrame(index=avail_cols, columns=avail_cols, dtype=float)
mae_matrix = pd.DataFrame(index=avail_cols, columns=avail_cols, dtype=float)
pbias_matrix = pd.DataFrame(index=avail_cols, columns=avail_cols, dtype=float)
kge_matrix = pd.DataFrame(index=avail_cols, columns=avail_cols, dtype=float)
nse_matrix = pd.DataFrame(index=avail_cols, columns=avail_cols, dtype=float)

for col_ref in avail_cols:
    for col_sim in avail_cols:
        mask = df_eval[col_ref].notna() & df_eval[col_sim].notna()
        ref = df_eval.loc[mask, col_ref].values
        sim = df_eval.loc[mask, col_sim].values
        
        if len(ref) < 10:
            continue
            
        rp, _ = stats.pearsonr(ref, sim)
        rs, _ = stats.spearmanr(ref, sim)
        rmse = np.sqrt(np.mean((sim - ref)**2))
        mae = np.mean(np.abs(sim - ref))
        bias = np.mean(sim - ref)
        pbias = 100.0 * np.sum(sim - ref) / np.sum(ref) if np.sum(ref) != 0 else 0
        
        # NSE
        denom_nse = np.sum((ref - np.mean(ref))**2)
        nse = 1 - (np.sum((ref - sim)**2) / denom_nse) if denom_nse != 0 else -999
        
        # KGE
        r_kge = rp
        alpha_kge = np.std(sim) / np.std(ref) if np.std(ref) != 0 else 1
        beta_kge = np.mean(sim) / np.mean(ref) if np.mean(ref) != 0 else 1
        kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha_kge - 1)**2 + (beta_kge - 1)**2)
        
        rmse_matrix.loc[col_ref, col_sim] = round(rmse, 2)
        mae_matrix.loc[col_ref, col_sim] = round(mae, 2)
        pbias_matrix.loc[col_ref, col_sim] = round(pbias, 1)
        kge_matrix.loc[col_ref, col_sim] = round(kge, 3)
        nse_matrix.loc[col_ref, col_sim] = round(nse, 3)
        
        if col_ref != col_sim:
            pairwise_all.append({
                'Reference (X)': col_ref,
                'Evaluation (Y)': col_sim,
                'Sampel (N)': len(ref),
                'Pearson r': round(rp, 3),
                'Spearman ρ': round(rs, 3),
                'R²': round(rp**2, 3),
                'RMSE (mm/hari)': round(rmse, 2),
                'MAE (mm/hari)': round(mae, 2),
                'Bias (mm/hari)': round(bias, 2),
                'PBIAS (%)': round(pbias, 1),
                'NSE': round(nse, 3),
                'KGE': round(kge, 3)
            })

df_pairwise_all = pd.DataFrame(pairwise_all)
df_pairwise_all.to_csv(os.path.join(out_dir, 'ringkasan_matriks_pairwise_all_to_all.csv'), index=False)

# Visualisasi 4 Heatmap Matriks Error (RMSE, MAE, KGE, PBIAS)
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Matriks Evaluasi Inter-Model All-to-All ($8 \\times 8$ Inter-Comparison Matrix)', fontsize=15, fontweight='bold', y=0.98)

# 1. RMSE Matrix
sns.heatmap(rmse_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0], linewidths=0.5, cbar_kws={'label': 'RMSE (mm/hari)'})
axes[0, 0].set_title('Matriks RMSE (mm/hari) [Baris = Acuan, Kolom = Model]', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=30)

# 2. MAE Matrix
sns.heatmap(mae_matrix, annot=True, fmt='.1f', cmap='Oranges', ax=axes[0, 1], linewidths=0.5, cbar_kws={'label': 'MAE (mm/hari)'})
axes[0, 1].set_title('Matriks MAE (mm/hari) [Baris = Acuan, Kolom = Model]', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=30)

# 3. KGE Matrix
sns.heatmap(kge_matrix, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-0.5, vmax=1.0, ax=axes[1, 0], linewidths=0.5, cbar_kws={'label': 'Kling-Gupta Efficiency (KGE)'})
axes[1, 0].set_title('Matriks KGE (Kling-Gupta Efficiency) [Optimal = 1.0]', fontsize=12, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=30)

# 4. PBIAS Matrix
sns.heatmap(pbias_matrix, annot=True, fmt='.0f', cmap='coolwarm', center=0, ax=axes[1, 1], linewidths=0.5, cbar_kws={'label': 'Percent Bias (PBIAS %)'})
axes[1, 1].set_title('Matriks PBIAS (%) [Nilai Positif = Overestimasi]', fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '02b_matriks_evaluasi_error_all_to_all.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 7. Evaluasi Khusus Benchmark CHIRPS Reanalisis (CHIRPS_RNL) ───
print("\nMembuat Evaluasi Benchmark terhadap CHIRPS Reanalisis (CHIRPS_RNL)...")
chirps_ref_cols = [c for c in avail_cols if c != 'CHIRPS_RNL']
n_comp = len(chirps_ref_cols)
ncols_sub = 4
nrows_sub = int(np.ceil(n_comp / ncols_sub))

fig, axes = plt.subplots(nrows_sub, ncols_sub, figsize=(5.5 * ncols_sub, 5.0 * nrows_sub), squeeze=False)
fig.suptitle('Scatter Plot & Hexbin Density vs CHIRPS Reanalisis (CHIRPS_RNL Benchmark, 2004 – 2025)\nEvaluasi Terhadap Data Gridded Reanalisis Kontinu 22 Tahun Lengkap',
             fontsize=14, fontweight='bold', y=1.02)

chirps_metrics = []

for idx, col_name in enumerate(chirps_ref_cols):
    row_idx = idx // ncols_sub
    col_idx = idx % ncols_sub
    ax = axes[row_idx][col_idx]
    
    mask = df_eval['CHIRPS_RNL'].notna() & df_eval[col_name].notna()
    x_ref = df_eval.loc[mask, 'CHIRPS_RNL'].values
    y_sim = df_eval.loc[mask, col_name].values
    
    n_samples = len(x_ref)
    rp, _ = stats.pearsonr(x_ref, y_sim)
    rs, _ = stats.spearmanr(x_ref, y_sim)
    slope, intercept, _, _, _ = stats.linregress(x_ref, y_sim)
    r2 = rp ** 2
    
    bias = np.mean(y_sim - x_ref)
    pbias = 100.0 * np.sum(y_sim - x_ref) / np.sum(x_ref) if np.sum(x_ref) != 0 else 0
    mae = np.mean(np.abs(y_sim - x_ref))
    rmse = np.sqrt(np.mean((y_sim - x_ref)**2))
    
    denom_nse = np.sum((x_ref - np.mean(x_ref))**2)
    nse = 1 - (np.sum((x_ref - y_sim)**2) / denom_nse) if denom_nse != 0 else -999
    
    r_kge = rp
    alpha_kge = np.std(y_sim) / np.std(x_ref) if np.std(x_ref) != 0 else 1
    beta_kge = np.mean(y_sim) / np.mean(x_ref) if np.mean(x_ref) != 0 else 1
    kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha_kge - 1)**2 + (beta_kge - 1)**2)
    
    chirps_metrics.append({
        'Produk': col_name,
        'Sampel (N)': n_samples,
        'Pearson r': round(rp, 3),
        'Spearman ρ': round(rs, 3),
        'R²': round(r2, 3),
        'RMSE (mm/hari)': round(rmse, 2),
        'MAE (mm/hari)': round(mae, 2),
        'Bias (mm/hari)': round(bias, 2),
        'PBIAS (%)': round(pbias, 1),
        'NSE': round(nse, 3),
        'KGE': round(kge, 3),
        'OLS Slope': round(slope, 3),
        'OLS Intercept': round(intercept, 3)
    })
    
    hb = ax.hexbin(x_ref, y_sim, gridsize=40, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('log₁₀(Count)', fontsize=7.5)
    
    jmax = max(x_ref.max(), y_sim.max())
    ax.plot([0, jmax], [0, jmax], 'k--', lw=1.3, alpha=0.7, label='1:1 Line')
    x_line = np.linspace(0, jmax, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', lw=1.8, label=f'OLS: y={slope:.2f}x+{intercept:.2f}')
    
    stats_box = (
        f"Pearson r = {rp:.3f}\n"
        f"Spearman ρ = {rs:.3f}\n"
        f"R² = {r2:.3f}\n"
        f"RMSE = {rmse:.2f} mm\n"
        f"MAE = {mae:.2f} mm\n"
        f"Bias = {bias:+.2f} mm ({pbias:+.1f}%)\n"
        f"NSE = {nse:.3f} | KGE = {kge:.3f}\n"
        f"N = {n_samples:,}"
    )
    ax.text(0.03, 0.97, stats_box, transform=ax.transAxes, fontsize=7.8, va='top', ha='left',
            family='monospace', bbox=dict(facecolor='white', alpha=0.88, edgecolor='gray', boxstyle='round,pad=0.35'))
    
    ax.set_title(f'{col_name} vs CHIRPS_RNL', fontsize=11, fontweight='bold')
    ax.set_xlabel('CHIRPS_RNL Benchmark (mm/hari)', fontsize=9)
    ax.set_ylabel(f'{col_name} (mm/hari)', fontsize=9)
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(True, linestyle=':', alpha=0.5)

for empty_idx in range(n_comp, nrows_sub * ncols_sub):
    axes[empty_idx // ncols_sub][empty_idx % ncols_sub].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '03b_scatter_hexbin_vs_chirps_rnl.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

df_chirps_metrics = pd.DataFrame(chirps_metrics)
df_chirps_metrics.to_csv(os.path.join(out_dir, 'ringkasan_metrik_evaluasi_vs_chirps_rnl.csv'), index=False)
print("\n=== RINGKASAN METRIK EVALUASI vs CHIRPS REANALISIS (CHIRPS_RNL) ===")
print(df_chirps_metrics.to_string(index=False))

# ─── 8. Bar Chart Perbandingan Akurasi vs CHIRPS_RNL ───
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Kinerja Akurasi Seluruh Produk Presipitasi terhadap CHIRPS Reanalisis Benchmark (2004 – 2025)',
             fontsize=14, fontweight='bold', y=0.98)

x_pos = np.arange(len(df_chirps_metrics))
width = 0.35

# 1. Pearson r & Spearman rho vs CHIRPS_RNL
ax1 = axes[0, 0]
ax1.bar(x_pos - width/2, df_chirps_metrics['Pearson r'], width, label='Pearson r', color='#2b83ba')
ax1.bar(x_pos + width/2, df_chirps_metrics['Spearman ρ'], width, label='Spearman ρ', color='#abdda4')
ax1.set_title('Koefisien Korelasi vs CHIRPS_RNL (r & ρ)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_chirps_metrics['Produk'], rotation=30, ha='right', fontsize=8.5)
ax1.set_ylim(0, 1.0)
ax1.axhline(0.7, color='green', linestyle='--', alpha=0.6, label='Korelasi Kuat (≥ 0.7)')
ax1.legend(fontsize=8.5)
for p in ax1.patches:
    h = p.get_height()
    if h > 0:
        ax1.annotate(f'{h:.2f}', (p.get_x() + p.get_width()/2, h + 0.01), ha='center', fontsize=7.5, fontweight='bold')

# 2. Error Metrics (RMSE & MAE) vs CHIRPS_RNL
ax2 = axes[0, 1]
# Filter ERA5_LAND raw untuk menjaga keterbacaan bar chart
df_plot_err = df_chirps_metrics.copy()
ax2.bar(x_pos - width/2, df_plot_err['RMSE (mm/hari)'], width, label='RMSE (mm/hari)', color='#d7191c')
ax2.bar(x_pos + width/2, df_plot_err['MAE (mm/hari)'], width, label='MAE (mm/hari)', color='#fdae61')
ax2.set_title('Indikator Error vs CHIRPS_RNL (RMSE & MAE)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df_plot_err['Produk'], rotation=30, ha='right', fontsize=8.5)
ax2.set_ylabel('mm/hari', fontsize=9.5)
ax2.legend(fontsize=8.5)
for p in ax2.patches:
    h = p.get_height()
    if h > 0 and h < 50:
        ax2.annotate(f'{h:.1f}', (p.get_x() + p.get_width()/2, h + 0.5), ha='center', fontsize=7.5, fontweight='bold')

# 3. Efisiensi Model (NSE & KGE) vs CHIRPS_RNL
ax3 = axes[1, 0]
ax3.bar(x_pos - width/2, df_chirps_metrics['NSE'], width, label='NSE (Nash-Sutcliffe)', color='#5e4fa2')
ax3.bar(x_pos + width/2, df_chirps_metrics['KGE'], width, label='KGE (Kling-Gupta)', color='#9e0142')
ax3.set_title('Efisiensi Hidrologis vs CHIRPS_RNL (NSE & KGE)', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(df_chirps_metrics['Produk'], rotation=30, ha='right', fontsize=8.5)
ax3.axhline(0, color='gray', linestyle='-', linewidth=0.8)
ax3.set_ylim(-1.0, 1.0)
ax3.legend(fontsize=8.5)
for p in ax3.patches:
    h = p.get_height()
    if h > -1.0 and h < 1.0:
        ax3.annotate(f'{h:.2f}', (p.get_x() + p.get_width()/2, h + (0.02 if h>=0 else -0.06)), ha='center', fontsize=7.5, fontweight='bold')

# 4. Percent Bias (PBIAS %) vs CHIRPS_RNL
ax4 = axes[1, 1]
colors_bias = ['#d7191c' if abs(b) > 50 else '#fdae61' if abs(b) > 25 else '#2b83ba' for b in df_chirps_metrics['PBIAS (%)']]
bars = ax4.bar(df_chirps_metrics['Produk'], df_chirps_metrics['PBIAS (%)'], color=colors_bias, width=0.55)
ax4.set_title('Percent Bias vs CHIRPS_RNL (PBIAS %)\n(Nilai Dekat 0 = Bias Volume Minimum)', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(df_chirps_metrics['Produk'], rotation=30, ha='right', fontsize=8.5)
ax4.set_ylabel('PBIAS (%)', fontsize=9.5)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
for bar, val in zip(bars, df_chirps_metrics['PBIAS (%)']):
    h = bar.get_height()
    if abs(h) < 150:
        ax4.annotate(f'{h:+.1f}%', (bar.get_x() + bar.get_width()/2, h + (2.0 if h>=0 else -6.0)), ha='center', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '04b_bar_akurasi_vs_chirps_rnl.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 9. Evaluasi Kategorikal Deteksi Hujan vs CHIRPS_RNL ───
print("\nMenghitung Metrik Kontingensi Deteksi Hujan vs CHIRPS_RNL...")
thresholds = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
contingency_chirps = []

for sat_name in chirps_ref_cols:
    mask = df_eval['CHIRPS_RNL'].notna() & df_eval[sat_name].notna()
    obs_vals = df_eval.loc[mask, 'CHIRPS_RNL'].values
    sim_vals = df_eval.loc[mask, sat_name].values
    
    for th in thresholds:
        obs_rain = obs_vals >= th
        sim_rain = sim_vals >= th
        
        hits = int(np.sum(obs_rain & sim_rain))
        misses = int(np.sum(obs_rain & ~sim_rain))
        false_alarms = int(np.sum(~obs_rain & sim_rain))
        correct_negatives = int(np.sum(~obs_rain & ~sim_rain))
        total = hits + misses + false_alarms + correct_negatives
        
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
        fbi = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else 0
        
        hss_denom = ((hits + misses)*(misses + correct_negatives) + (hits + false_alarms)*(false_alarms + correct_negatives))
        hss = 2 * (hits * correct_negatives - misses * false_alarms) / hss_denom if hss_denom > 0 else 0
        
        hits_random = (hits + misses) * (hits + false_alarms) / total if total > 0 else 0
        ets_denom = hits + misses + false_alarms - hits_random
        ets = (hits - hits_random) / ets_denom if ets_denom > 0 else 0
        
        contingency_chirps.append({
            'Acuan Benchmark': 'CHIRPS_RNL',
            'Produk Presipitasi': sat_name,
            'Ambang Batas (mm/hari)': th,
            'Hits': hits,
            'Misses': misses,
            'False Alarms': false_alarms,
            'Correct Negatives': correct_negatives,
            'POD (Hit Rate)': round(pod, 3),
            'FAR (False Alarm)': round(far, 3),
            'CSI (Threat Score)': round(csi, 3),
            'FBI (Frequency Bias)': round(fbi, 3),
            'HSS (Heidke Skill)': round(hss, 3),
            'ETS (Equitable Threat)': round(ets, 3)
        })

df_contingency_chirps = pd.DataFrame(contingency_chirps)
df_contingency_chirps.to_csv(os.path.join(out_dir, 'ringkasan_metrik_kategorikal_vs_chirps_rnl.csv'), index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Kinerja Deteksi Kejadian Hujan terhadap CHIRPS Reanalisis Benchmark Berdasarkan Ambang Batas (mm/hari)',
             fontsize=14, fontweight='bold', y=0.98)

palette = {
    'CHIRPS_SAT': '#377eb8', 'GSMaP': '#4daf4a', 'IMERG': '#984ea3',
    'PERSIANN': '#ff7f00', 'ERA5': '#a65628', 'ERA5_LAND': '#f781bf', 'OYA': '#e41a1c'
}

for ax, metric, title, ylim in zip(
    axes.flatten(),
    ['CSI (Threat Score)', 'POD (Hit Rate)', 'FAR (False Alarm)', 'HSS (Heidke Skill)'],
    ['Critical Success Index (CSI / Threat Score) [Lebih tinggi lebih baik]',
     'Probability of Detection (POD / Hit Rate) [Lebih tinggi lebih baik]',
     'False Alarm Ratio (FAR) [Lebih rendah lebih baik]',
     'Heidke Skill Score (HSS) [Lebih tinggi lebih baik]'],
    [(0, 0.9), (0, 1.05), (0, 0.9), (0, 0.85)]
):
    for sat_name in chirps_ref_cols:
        sub = df_contingency_chirps[df_contingency_chirps['Produk Presipitasi'] == sat_name]
        ax.plot(sub['Ambang Batas (mm/hari)'], sub[metric], marker='o', lw=1.8, label=sat_name, color=palette.get(sat_name))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Ambang Batas Curah Hujan (mm/hari)', fontsize=9.5)
    ax.set_ylabel(metric, fontsize=9.5)
    ax.set_xscale('log')
    ax.set_xticks(thresholds)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '05b_kategorikal_skill_vs_chirps_rnl.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 10. Analisis Tren Temporal Jangka Panjang (2004 – 2025) ───
print("\nMembuat Visualisasi Time Series & Tren Jangka Panjang (2004 – 2025)...")

df['year'] = df.index.year
yearly_rain = df.groupby('year')[avail_cols].sum()
yearly_rain.to_csv(os.path.join(out_dir, 'ringkasan_total_tahunan_2004_2025.csv'))

fig, ax = plt.subplots(figsize=(15, 6.5))
for col in ['CHIRPS_RNL', 'CHIRPS_SAT', 'GSMaP', 'IMERG', 'PERSIANN', 'ERA5', 'OYA']:
    if col in yearly_rain.columns:
        ls = '-' if col == 'CHIRPS_RNL' else ('--' if col == 'OYA' else '-.')
        lw = 2.5 if col in ['CHIRPS_RNL', 'IMERG', 'OYA'] else 1.6
        marker = 'D' if col == 'CHIRPS_RNL' else ('o' if col == 'OYA' else 's')
        ax.plot(yearly_rain.index, yearly_rain[col], label=f'{col}', lw=lw, linestyle=ls, marker=marker, markersize=4.5)

ax.set_title('Perbandingan Akumulasi Curah Hujan Tahunan (2004 – 2025) di Pos Hujan Oya dan Sekitarnya\n(CHIRPS Reanalisis sebagai Garis Acuan Solid Tebal)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Tahun', fontsize=11)
ax.set_ylabel('Total Curah Hujan Tahunan (mm/tahun)', fontsize=11)
ax.set_xticks(yearly_rain.index)
ax.set_xticklabels(yearly_rain.index, rotation=45, fontsize=9)
ax.legend(loc='upper right', fontsize=8.5, ncol=2)
ax.grid(True, linestyle=':', alpha=0.6)

# Anotasi fenomena ekstrem
ax.annotate('La Niña Kuat 2010', xy=(2010, yearly_rain.loc[2010, 'CHIRPS_RNL']), xytext=(2008, 4800),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5), fontsize=9, fontweight='bold', color='blue')
ax.annotate('La Niña Kuat 2016', xy=(2016, yearly_rain.loc[2016, 'CHIRPS_RNL']), xytext=(2014, 4600),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5), fontsize=9, fontweight='bold', color='blue')
ax.annotate('El Niño Kuat 2015', xy=(2015, yearly_rain.loc[2015, 'CHIRPS_RNL']), xytext=(2013, 1400),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9, fontweight='bold', color='red')
ax.annotate('El Niño Kuat 2023', xy=(2023, yearly_rain.loc[2023, 'CHIRPS_RNL']), xytext=(2021, 1200),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '06_tren_akumulasi_hujan_tahunan_2004_2025.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# 2. Moving Average 30-Hari Time Series
df_rolling_30d = df[avail_cols].rolling(window=30, min_periods=5).mean()

fig, ax = plt.subplots(figsize=(16, 5))
for col in ['CHIRPS_RNL', 'IMERG', 'CHIRPS_SAT', 'GSMaP', 'ERA5', 'OYA']:
    if col in df_rolling_30d.columns:
        lw = 2.2 if col in ['CHIRPS_RNL', 'IMERG', 'OYA'] else 1.2
        alpha = 0.9 if col in ['CHIRPS_RNL', 'OYA'] else 0.65
        ax.plot(df_rolling_30d.index, df_rolling_30d[col], label=f'{col} (30-day MA)', lw=lw, alpha=alpha)

ax.set_title('Dinamika Fluktuasi Curah Hujan Rata-Rata 30-Hari (30-Day Moving Average, 2004 – 2025)\n(Membandingkan Siklus Intra-Musiman terhadap CHIRPS Reanalisis)',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Tahun', fontsize=11)
ax.set_ylabel('Rata-Rata Hujan 30-Hari (mm/hari)', fontsize=11)
ax.legend(loc='upper right', fontsize=9.0)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '07_timeseries_30day_moving_average.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 11. Kurva Massa Ganda (Double-Mass Curve terhadap CHIRPS_RNL) ───
print("Membuat Kurva Massa Ganda terhadap CHIRPS_RNL Benchmark...")
df_sorted = df[avail_cols].sort_index()
cum_chirps = df_sorted['CHIRPS_RNL'].cumsum() / 1000.0

fig, ax = plt.subplots(figsize=(11, 8))
for col_name in ['CHIRPS_SAT', 'GSMaP', 'IMERG', 'PERSIANN', 'ERA5', 'OYA']:
    if col_name in df_sorted.columns:
        cum_sim = df_sorted[col_name].cumsum() / 1000.0
        ax.plot(cum_chirps, cum_sim, label=f'{col_name}', lw=2.0)

max_cum = cum_chirps.max()
ax.plot([0, max_cum], [0, max_cum], 'k--', lw=1.5, alpha=0.7, label='1:1 Identik')

ax.set_title('Kurva Massa Ganda (Double-Mass Curve): Akumulasi Model vs CHIRPS Reanalisis Benchmark\n(Evaluasi Homogenitas & Konsistensi Jangka Panjang 2004 – 2025)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Akumulasi Curah Hujan CHIRPS_RNL Benchmark (Meter / 1.000 mm)', fontsize=11)
ax.set_ylabel('Akumulasi Curah Hujan Produk Evaluasi (Meter / 1.000 mm)', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '08_kurva_massa_ganda_double_mass_curve.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 12. Pola Musiman & Klimatologi Rata-Rata Bulanan (Jan – Des) ───
print("Membuat Pola Musiman & Klimatologi Bulanan...")
df['month'] = df.index.month
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Total bulanan rata-rata (mm/bulan)
monthly_mean_daily = df.groupby('month')[avail_cols].mean()
monthly_clim = monthly_mean_daily.copy()
for c in avail_cols:
    monthly_clim[c] = monthly_clim[c] * 30.4375

monthly_clim.index = month_order
monthly_clim.to_csv(os.path.join(out_dir, 'ringkasan_klimatologi_bulanan_2004_2025.csv'))

# Bar chart klimatologi
fig, ax = plt.subplots(figsize=(15, 6.5))
clim_plot_cols = [c for c in avail_cols if c != 'ERA5_LAND']
df_clim_melted = monthly_clim[clim_plot_cols].reset_index().rename(columns={'index': 'Bulan'}).melt(id_vars='Bulan', var_name='Dataset', value_name='Curah_Hujan_mm')
df_clim_melted['Bulan'] = pd.Categorical(df_clim_melted['Bulan'], categories=month_order, ordered=True)

sns.barplot(data=df_clim_melted, x='Bulan', y='Curah_Hujan_mm', hue='Dataset', palette='tab10', ax=ax)
ax.set_title('Pola Musiman Rata-Rata Curah Hujan Bulanan (Klimatologi 2004 – 2025)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Bulan', fontsize=11)
ax.set_ylabel('Total Curah Hujan Rata-Rata (mm/bulan)', fontsize=11)
ax.legend(loc='upper right', fontsize=8.5, ncol=2)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '09_klimatologi_bulanan_barchart.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Boxplot Variabilitas Bulanan
df_monthly_totals = df[clim_plot_cols].resample('ME').sum()
df_monthly_totals['month_name'] = df_monthly_totals.index.strftime('%b')
df_monthly_melted = df_monthly_totals.melt(id_vars='month_name', value_vars=clim_plot_cols, var_name='Dataset', value_name='Total_Hujan_Bulanan_mm')
df_monthly_melted['month_name'] = pd.Categorical(df_monthly_melted['month_name'], categories=month_order, ordered=True)

fig, ax = plt.subplots(figsize=(16, 6.5))
sns.boxplot(data=df_monthly_melted, x='month_name', y='Total_Hujan_Bulanan_mm', hue='Dataset', palette='tab10',
            showfliers=True, fliersize=2.2, linewidth=1.0, ax=ax)
ax.set_title('Diagram Kotak (Boxplot) Variabilitas & Sebaran Curah Hujan Bulanan (2004 – 2025)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Bulan', fontsize=11)
ax.set_ylabel('Curah Hujan Bulanan (mm/bulan)', fontsize=11)
ax.legend(loc='upper right', fontsize=8.5, ncol=2)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '10_boxplot_variabilitas_bulanan.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 13. Distribusi Kumulatif Intensitas Hujan (CDF & PDF) ───
print("Membuat Kurva Probabilitas Kumulatif Intensitas Hujan (CDF & PDF)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for col in clim_plot_cols:
    rain_wet = df[col].dropna()
    rain_wet = rain_wet[rain_wet > 0.1]
    sns.kdeplot(rain_wet, label=col, lw=1.8, ax=ax1, log_scale=True)

ax1.set_title('Distribusi Kepadatan Probabilitas (PDF)\nIntensitas Hujan Hari Basah (> 0.1 mm/hari)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Curah Hujan (mm/hari, skala log)', fontsize=10)
ax1.set_ylabel('Kepadatan Densitas (KDE)', fontsize=10)
ax1.legend(fontsize=8.5)
ax1.grid(True, which='both', linestyle=':', alpha=0.6)

for col in clim_plot_cols:
    rain_wet = df[col].dropna()
    rain_wet = rain_wet[rain_wet > 0.1].sort_values()
    prob = np.arange(1, len(rain_wet) + 1) / len(rain_wet)
    ax2.plot(rain_wet, 1.0 - prob, label=col, lw=1.8)

ax2.set_title('Kurva Probabilitas Pelampauan (Exceedance Probability / CDF)\nFrekuensi Terjadinya Hujan Melebihi Intensitas Tertentu', fontsize=12, fontweight='bold')
ax2.set_xlabel('Ambang Batas Intensitas Curah Hujan (mm/hari)', fontsize=10)
ax2.set_ylabel('P(Hujan ≥ x)', fontsize=10)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(fontsize=8.5)
ax2.grid(True, which='both', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '11_distribusi_pdf_cdf_intensitas_hujan.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 14. Indeks Anomali Curah Hujan Bulanan (CHIRPS_RNL vs GPM IMERG) ───
print("Membuat Grafik Tren Anomali Curah Hujan CHIRPS_RNL...")
monthly_chirps = df['CHIRPS_RNL'].resample('MS').sum()
clim_chirps_m = monthly_chirps.groupby(monthly_chirps.index.month).transform('mean')
anom_chirps = monthly_chirps - clim_chirps_m

fig, ax = plt.subplots(figsize=(15, 5))
ax.fill_between(anom_chirps.index, anom_chirps.values, 0, where=(anom_chirps.values >= 0), color='#2b83ba', alpha=0.75, label='Surplus Hujan (Wet Anomaly / La Niña)')
ax.fill_between(anom_chirps.index, anom_chirps.values, 0, where=(anom_chirps.values < 0), color='#d7191c', alpha=0.75, label='Defisit Hujan (Dry Anomaly / El Niño)')
ax.plot(anom_chirps.index, anom_chirps.values, color='black', lw=0.6, alpha=0.4)

# Linear trendline
x_sec = (anom_chirps.index - anom_chirps.index[0]).total_seconds() / (86400 * 365.25)
slope_anom, int_anom = np.polyfit(x_sec, anom_chirps.values, 1)
ax.plot(anom_chirps.index, slope_anom * x_sec + int_anom, color='darkblue', linestyle='--', lw=2,
        label=f'Tren Anomali ({slope_anom:+.2f} mm/bln/thn)')

ax.axhline(0, color='gray', linestyle='-', lw=1.0)
ax.set_title('Indeks Anomali Curah Hujan Bulanan CHIRPS Reanalisis Benchmark (2004 – 2025)\n(Deviasi Terhadap Rata-Rata Klimatologi Bulanan Historis 22 Tahun)', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Tahun', fontsize=11)
ax.set_ylabel('Anomali Presipitasi (mm/bulan)', fontsize=11)
ax.legend(loc='upper right', fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '12_tren_anomali_curah_hujan_bulanan.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ─── 15. Uji Signifikansi Statistik Non-Parametrik Lengkap (28 Pasangan All-to-All) ───
print("\nMenjalankan Uji Signifikansi Statistik Lengkap (Wilcoxon Signed-Rank untuk Semua Pasangan)...")

# 1. Wilcoxon Signed-Rank Test untuk Seluruh Pasangan Data Harian
wilcoxon_all = []
for i, col_a in enumerate(avail_cols):
    for j, col_b in enumerate(avail_cols):
        if j <= i: continue
        sub = df_eval[[col_a, col_b]].dropna()
        wet_sub = sub[(sub[col_a] >= 0.1) | (sub[col_b] >= 0.1)]
        if len(wet_sub) > 10:
            stat_w, p_val_w = stats.wilcoxon(wet_sub[col_a], wet_sub[col_b])
            sig_w = "Ya (p < 0.05)" if p_val_w < 0.05 else "Tidak (p ≥ 0.05)"
            wilcoxon_all.append({
                'Pasangan Inter-Comparison': f'{col_a} ↔ {col_b}',
                'Jumlah Sampel Hari Basah': len(wet_sub),
                f'Median {col_a} (mm)': round(float(np.median(wet_sub[col_a])), 2),
                f'Median {col_b} (mm)': round(float(np.median(wet_sub[col_b])), 2),
                'Statistik W': round(stat_w, 1),
                'p-value': f"{p_val_w:.4e}",
                'Berbeda Nyata Signifikan?': sig_w
            })

df_wilcoxon_all = pd.DataFrame(wilcoxon_all)
df_wilcoxon_all.to_csv(os.path.join(out_dir, 'ringkasan_uji_wilcoxon_all_pairs.csv'), index=False)
print("\n=== HASIL UJI WILCOXON SIGNED-RANK (SEMUA PASANGAN INTER-COMPARISON) ===")
print(df_wilcoxon_all.head(15).to_string(index=False))

# 2. Mann-Whitney U Test (Unpaired: Musim Hujan Nov–Apr vs Musim Kemarau Mei–Okt)
wet_months = [11, 12, 1, 2, 3, 4]
dry_months = [5, 6, 7, 8, 9, 10]

mw_rows = []
for col in avail_cols:
    wet_data = df[df['month'].isin(wet_months)][col].dropna()
    dry_data = df[df['month'].isin(dry_months)][col].dropna()
    
    if len(wet_data) > 10 and len(dry_data) > 10:
        stat_mw, p_val_mw = stats.mannwhitneyu(wet_data, dry_data, alternative='two-sided')
        sig_mw = "Ya (p < 0.05)" if p_val_mw < 0.05 else "Tidak (p ≥ 0.05)"
        mw_rows.append({
            'Dataset': col,
            'N Musim Hujan (Nov-Apr)': len(wet_data),
            'Median Musim Hujan (mm/hari)': round(float(np.median(wet_data)), 2),
            'N Musim Kemarau (Mei-Okt)': len(dry_data),
            'Median Musim Kemarau (mm/hari)': round(float(np.median(dry_data)), 2),
            'Statistik U': round(stat_mw, 1),
            'p-value': f"{p_val_mw:.4e}",
            'Perbedaan Musiman Signifikan?': sig_mw
        })

df_mw = pd.DataFrame(mw_rows)
df_mw.to_csv(os.path.join(out_dir, 'ringkasan_uji_mann_whitney_musiman.csv'), index=False)

# Simpan semua ringkasan ke file Excel multi-sheet
with pd.ExcelWriter(os.path.join(out_dir, 'ringkasan_komparasi_inter_model_all_to_all.xlsx'), engine='openpyxl') as writer:
    df_pairwise_all.to_excel(writer, sheet_name='All_to_All_Pairwise', index=False)
    df_chirps_metrics.to_excel(writer, sheet_name='Benchmark_vs_CHIRPS_RNL', index=False)
    df_contingency_chirps.to_excel(writer, sheet_name='Kategorikal_vs_CHIRPS', index=False)
    df_wilcoxon_all.to_excel(writer, sheet_name='Wilcoxon_All_Pairs', index=False)
    df_mw.to_excel(writer, sheet_name='Mann_Whitney_Seasonal', index=False)
    desc_stats.to_excel(writer, sheet_name='Statistik_Deskriptif')

print(f"\n🎉 Seluruh analisis All-to-All Inter-Comparison Matrix dan Benchmark CHIRPS_RNL berhasil diekspor ke: {out_dir}")
