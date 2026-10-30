"""
Script: generate_etccdi_plots.py
Menghasilkan 9 Panel Grafik Publikasi Resolusi Tinggi (300 DPI)
untuk 27 Indeks Iklim Ekstrem ETCCDI Kabupaten Kebumen (2000-2025)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup paths
BASE_DIR = Path(r"d:\Github\Projek_Rainfall")
OUT_DIR = BASE_DIR / "Google_Earth_Engine" / "Dokumen Analisis" / "Analisis ETCCDI Kebumen"
TABLES_DIR = OUT_DIR / "tables"
FIGS_DIR = OUT_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Styling for scientific papers
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})

# Load datasets
df_etccdi = pd.read_csv(TABLES_DIR / "etccdi_indices_annual_2000_2025.csv")
df_summary = pd.read_csv(TABLES_DIR / "etccdi_summary_statistics_trends.csv")
df_sat = pd.read_csv(TABLES_DIR / "satellite_precipitation_indices_comparison.csv")

years = df_etccdi['year'].values

def add_trendline(ax, x, y, color='crimson', label_pos='top', unit=''):
    """Helper to draw Sen's slope trend line with MK significance annotation."""
    res_slope = stats.theilslopes(y, x)
    slope, intercept = res_slope[0], res_slope[1]
    y_pred = intercept + slope * x
    
    # MK Z and p-value
    n = len(x)
    s = sum(np.sign(y[j] - y[k]) for k in range(n-1) for j in range(k+1, n))
    unique_y, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    z = (s - 1)/np.sqrt(var_s) if s > 0 else ((s + 1)/np.sqrt(var_s) if s < 0 else 0)
    pval = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    
    ax.plot(x, y_pred, color=color, linestyle='--', linewidth=2, label=f"Sen's Slope: {slope*10:+.2f} {unit}/dekade\n(p = {pval:.4f})")
    return slope, pval

# =============================================================================
# FIG 1: TEMPERATURE EXTREMES ABSOLUT (TXx, TXn, TNx, TNn)
# =============================================================================
print("Generating Fig 1: Temperature Extremes Absolut...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# TXx
ax = axes[0, 0]
ax.plot(years, df_etccdi['TXx'], marker='o', color='#d95f02', linewidth=1.5, label='TXx Tahunan')
ax.plot(years, df_etccdi['TXx'].rolling(5, min_periods=1).mean(), color='#7570b3', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['TXx'].values, color='#e7298a', unit='°C')
ax.set_title('(a) TXx: Suhu Maksimum Tertinggi Tahunan')
ax.set_ylabel('Suhu (°C)')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# TXn
ax = axes[0, 1]
ax.plot(years, df_etccdi['TXn'], marker='s', color='#1b9e77', linewidth=1.5, label='TXn Tahunan')
ax.plot(years, df_etccdi['TXn'].rolling(5, min_periods=1).mean(), color='#7570b3', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['TXn'].values, color='#e7298a', unit='°C')
ax.set_title('(b) TXn: Suhu Maksimum Terendah Tahunan')
ax.set_ylabel('Suhu (°C)')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# TNx
ax = axes[1, 0]
ax.plot(years, df_etccdi['TNx'], marker='^', color='#e6ab02', linewidth=1.5, label='TNx Tahunan')
ax.plot(years, df_etccdi['TNx'].rolling(5, min_periods=1).mean(), color='#7570b3', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['TNx'].values, color='crimson', unit='°C')
ax.set_title('(c) TNx: Suhu Minimum Tertinggi Tahunan (Malam Terpanas)')
ax.set_ylabel('Suhu (°C)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# TNn
ax = axes[1, 1]
ax.plot(years, df_etccdi['TNn'], marker='d', color='#386cb0', linewidth=1.5, label='TNn Tahunan')
ax.plot(years, df_etccdi['TNn'].rolling(5, min_periods=1).mean(), color='#7570b3', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['TNn'].values, color='#e7298a', unit='°C')
ax.set_title('(d) TNn: Suhu Minimum Terendah Tahunan (Malam Terdingin)')
ax.set_ylabel('Suhu (°C)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig1_temperature_extremes_absolut.png", dpi=300)
plt.close()

# =============================================================================
# FIG 2: TEMPERATURE PERCENTILES (TX10p, TX90p, TN10p, TN90p)
# =============================================================================
print("Generating Fig 2: Temperature Percentiles...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# TX10p
ax = axes[0, 0]
ax.bar(years, df_etccdi['TX10p'], color='#4575b4', alpha=0.7, width=0.6, label='TX10p Tahunan')
add_trendline(ax, years, df_etccdi['TX10p'].values, color='#08519c', unit='%')
ax.axhline(10.0, color='gray', linestyle=':', label='Garis Acuan Baseline (10%)')
ax.set_title('(a) TX10p: Hari Dingin Siang Hari (Cool Days)')
ax.set_ylabel('Persentase Hari (%)')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# TX90p
ax = axes[0, 1]
ax.bar(years, df_etccdi['TX90p'], color='#d73027', alpha=0.7, width=0.6, label='TX90p Tahunan')
add_trendline(ax, years, df_etccdi['TX90p'].values, color='#a50026', unit='%')
ax.axhline(10.0, color='gray', linestyle=':', label='Garis Acuan Baseline (10%)')
ax.set_title('(b) TX90p: Hari Panas Siang Hari (Warm Days)')
ax.set_ylabel('Persentase Hari (%)')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# TN10p
ax = axes[1, 0]
ax.bar(years, df_etccdi['TN10p'], color='#313695', alpha=0.7, width=0.6, label='TN10p Tahunan')
add_trendline(ax, years, df_etccdi['TN10p'].values, color='#08306b', unit='%')
ax.axhline(10.0, color='gray', linestyle=':', label='Garis Acuan Baseline (10%)')
ax.set_title('(c) TN10p: Malam Dingin (Cool Nights)')
ax.set_ylabel('Persentase Hari (%)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# TN90p
ax = axes[1, 1]
ax.bar(years, df_etccdi['TN90p'], color='#f46d43', alpha=0.7, width=0.6, label='TN90p Tahunan')
add_trendline(ax, years, df_etccdi['TN90p'].values, color='#67000d', unit='%')
ax.axhline(10.0, color='gray', linestyle=':', label='Garis Acuan Baseline (10%)')
ax.set_title('(d) TN90p: Malam Terik / Hangat (Warm Nights)')
ax.set_ylabel('Persentase Hari (%)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig2_temperature_percentiles.png", dpi=300)
plt.close()

# =============================================================================
# FIG 3: SPELLS AND DTR (WSDI, CSDI, DTR, TR)
# =============================================================================
print("Generating Fig 3: Spells and Diurnal Range...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# WSDI
ax = axes[0, 0]
ax.bar(years, df_etccdi['WSDI'], color='#e41a1c', alpha=0.75, width=0.6, label='WSDI Tahunan')
add_trendline(ax, years, df_etccdi['WSDI'].values, color='#800026', unit='hari')
ax.set_title('(a) WSDI: Indeks Durasi Gelombang Panas (Warm Spell Duration)')
ax.set_ylabel('Jumlah Hari')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# CSDI
ax = axes[0, 1]
ax.bar(years, df_etccdi['CSDI'], color='#377eb8', alpha=0.75, width=0.6, label='CSDI Tahunan')
add_trendline(ax, years, df_etccdi['CSDI'].values, color='#08519c', unit='hari')
ax.set_title('(b) CSDI: Indeks Durasi Gelombang Dingin (Cold Spell Duration)')
ax.set_ylabel('Jumlah Hari')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# DTR
ax = axes[1, 0]
ax.plot(years, df_etccdi['DTR'], marker='o', color='#4daf4a', linewidth=1.8, label='DTR Tahunan')
ax.plot(years, df_etccdi['DTR'].rolling(5, min_periods=1).mean(), color='#984ea3', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['DTR'].values, color='#006d2c', unit='°C')
ax.set_title('(c) DTR: Rentang Suhu Diurnal Harian (TX - TN)')
ax.set_ylabel('Rentang Suhu (°C)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# TR
ax = axes[1, 1]
ax.plot(years, df_etccdi['TR'], marker='s', color='#ff7f00', linewidth=1.8, label='TR Tahunan (TN > 20°C)')
add_trendline(ax, years, df_etccdi['TR'].values, color='crimson', unit='hari')
ax.set_title('(d) TR: Frekuensi Malam Tropis (Tropical Nights)')
ax.set_ylabel('Jumlah Hari / Tahun')
ax.set_xlabel('Tahun')
ax.legend(loc='lower left', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig3_spells_and_dtr.png", dpi=300)
plt.close()

# =============================================================================
# FIG 4: PRECIPITATION MAXIMA (Rx1day, Rx5day)
# =============================================================================
print("Generating Fig 4: Precipitation Maxima...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

# Rx1day
ax = axes[0]
ax.plot(years, df_etccdi['Rx1day'], marker='o', color='#1f78b4', linewidth=1.8, label='Rx1day Tahunan')
ax.plot(years, df_etccdi['Rx1day'].rolling(5, min_periods=1).mean(), color='#33a02c', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['Rx1day'].values, color='#e31a1c', unit='mm')
# Annotate extreme max
max_idx = df_etccdi['Rx1day'].idxmax()
ax.scatter([years[max_idx]], [df_etccdi['Rx1day'].iloc[max_idx]], color='red', s=80, zorder=5)
ax.annotate(f"Puncak: {df_etccdi['Rx1day'].iloc[max_idx]:.1f} mm\n({years[max_idx]})",
            (years[max_idx], df_etccdi['Rx1day'].iloc[max_idx]),
            textcoords="offset points", xytext=(-20, 10), ha='center',
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
ax.set_title('(a) Rx1day: Curah Hujan Harian Maksimum 1 Hari')
ax.set_ylabel('Presipitasi (mm/hari)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# Rx5day
ax = axes[1]
ax.plot(years, df_etccdi['Rx5day'], marker='s', color='#2ca25f', linewidth=1.8, label='Rx5day Tahunan')
ax.plot(years, df_etccdi['Rx5day'].rolling(5, min_periods=1).mean(), color='#ff7f00', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['Rx5day'].values, color='#e31a1c', unit='mm')
max5_idx = df_etccdi['Rx5day'].idxmax()
ax.scatter([years[max5_idx]], [df_etccdi['Rx5day'].iloc[max5_idx]], color='red', s=80, zorder=5)
ax.annotate(f"Puncak: {df_etccdi['Rx5day'].iloc[max5_idx]:.1f} mm\n({years[max5_idx]})",
            (years[max5_idx], df_etccdi['Rx5day'].iloc[max5_idx]),
            textcoords="offset points", xytext=(-20, 10), ha='center',
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
ax.set_title('(b) Rx5day: Akumulasi Maksimum 5 Hari Beruntun')
ax.set_ylabel('Presipitasi (mm / 5 hari)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig4_precipitation_maxima.png", dpi=300)
plt.close()

# =============================================================================
# FIG 5: PRECIPITATION TOTALS AND INTENSITY (PRCPTOT, SDII)
# =============================================================================
print("Generating Fig 5: Precipitation Totals & Intensity...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

# PRCPTOT
ax = axes[0]
bars = ax.bar(years, df_etccdi['PRCPTOT'], color='#43a2ca', alpha=0.75, width=0.65, label='PRCPTOT Tahunan')
add_trendline(ax, years, df_etccdi['PRCPTOT'].values, color='#08519c', unit='mm')
mean_prcp = df_etccdi['PRCPTOT'].mean()
ax.axhline(mean_prcp, color='black', linestyle=':', label=f'Rerata 26-Tahun ({mean_prcp:.0f} mm)')
ax.set_title('(a) PRCPTOT: Total Curah Hujan Tahunan Hari Basah (>= 1.0 mm)')
ax.set_ylabel('Total Presipitasi (mm)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# SDII
ax = axes[1]
ax.plot(years, df_etccdi['SDII'], marker='o', color='#8856a7', linewidth=1.8, label='SDII Tahunan')
ax.plot(years, df_etccdi['SDII'].rolling(5, min_periods=1).mean(), color='#e7298a', linewidth=2, label='Rerata Bergerak 5-Tahun')
add_trendline(ax, years, df_etccdi['SDII'].values, color='#49006a', unit='mm/hari')
mean_sdii = df_etccdi['SDII'].mean()
ax.axhline(mean_sdii, color='black', linestyle=':', label=f'Rerata 26-Tahun ({mean_sdii:.2f} mm/hari)')
ax.set_title('(b) SDII: Simple Daily Intensity Index (Intensitas Harian)')
ax.set_ylabel('Intensitas (mm/hari)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig5_precipitation_totals_and_intensity.png", dpi=300)
plt.close()

# =============================================================================
# FIG 6: EXTREME PRECIPITATION DAYS (R10mm, R20mm, Rnnmm)
# =============================================================================
print("Generating Fig 6: Extreme Precipitation Days...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

# R10mm
ax = axes[0]
ax.bar(years, df_etccdi['R10mm'], color='#7bccc4', width=0.65, label='R10mm')
add_trendline(ax, years, df_etccdi['R10mm'].values, color='#08589e', unit='hari')
ax.set_title('(a) R10mm: Hari Hujan Lebat (>= 10 mm)')
ax.set_ylabel('Jumlah Hari')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# R20mm
ax = axes[1]
ax.bar(years, df_etccdi['R20mm'], color='#4eb3d3', width=0.65, label='R20mm')
add_trendline(ax, years, df_etccdi['R20mm'].values, color='#08589e', unit='hari')
ax.set_title('(b) R20mm: Hari Hujan Sangat Lebat (>= 20 mm)')
ax.set_ylabel('Jumlah Hari')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# R50mm
ax = axes[2]
ax.bar(years, df_etccdi['Rnnmm'], color='#2b8cbe', width=0.65, label='Rnnmm (nn=50)')
add_trendline(ax, years, df_etccdi['Rnnmm'].values, color='crimson', unit='hari')
ax.set_title('(c) Rnnmm: Hari Hujan Ekstrem (nn=50 mm)')
ax.set_ylabel('Jumlah Hari')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig6_extreme_precipitation_days.png", dpi=300)
plt.close()

# =============================================================================
# FIG 7: DRY AND WET SPELLS (CDD, CWD)
# =============================================================================
print("Generating Fig 7: Dry and Wet Spells...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

# CDD
ax = axes[0]
ax.bar(years, df_etccdi['CDD'], color='#feb24c', alpha=0.8, width=0.65, label='CDD Tahunan')
add_trendline(ax, years, df_etccdi['CDD'].values, color='#bd0026', unit='hari')
ax.set_title('(a) CDD: Hari Kering Berturut-turut Terpanjang (Consecutive Dry Days)')
ax.set_ylabel('Durasi (Hari)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

# CWD
ax = axes[1]
ax.bar(years, df_etccdi['CWD'], color='#67a9cf', alpha=0.8, width=0.65, label='CWD Tahunan')
add_trendline(ax, years, df_etccdi['CWD'].values, color='#02818a', unit='hari')
ax.set_title('(b) CWD: Hari Basah Berturut-turut Terpanjang (Consecutive Wet Days)')
ax.set_ylabel('Durasi (Hari)')
ax.set_xlabel('Tahun')
ax.legend(loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig7_dry_wet_spells.png", dpi=300)
plt.close()

# =============================================================================
# FIG 8: MULTI-SATELLITE INTERCOMPARISON (Rx1day, SDII, PRCPTOT, Rnnmm)
# =============================================================================
print("Generating Fig 8: Satellite Intercomparison...")
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
prod_colors = {'ERA5_LAND': '#1f77b4', 'CHIRPS_RNL': '#2ca02c', 'GSMaP': '#ff7f0e', 'IMERG': '#d62728'}

# (a) Rx1day Boxplot
ax = axes[0, 0]
sns.boxplot(data=df_sat, x='Product', y='Rx1day', hue='Product', palette=prod_colors, legend=False, ax=ax, width=0.5)
ax.set_title('(a) Distribusi Curah Hujan Harian Maksimum (Rx1day)')
ax.set_ylabel('Rx1day (mm/hari)')
ax.set_xlabel('')

# (b) SDII Boxplot
ax = axes[0, 1]
sns.boxplot(data=df_sat, x='Product', y='SDII', hue='Product', palette=prod_colors, legend=False, ax=ax, width=0.5)
ax.set_title('(b) Distribusi Intensitas Harian Sederhana (SDII)')
ax.set_ylabel('SDII (mm/hari)')
ax.set_xlabel('')

# (c) PRCPTOT Boxplot
ax = axes[1, 0]
sns.boxplot(data=df_sat, x='Product', y='PRCPTOT', hue='Product', palette=prod_colors, legend=False, ax=ax, width=0.5)
ax.set_title('(c) Total Presipitasi Tahunan Hari Basah (PRCPTOT)')
ax.set_ylabel('PRCPTOT (mm/tahun)')
ax.set_xlabel('Produk Satelit / Reanalisis')

# (d) R50mm Bar Mean
ax = axes[1, 1]
mean_rnn = df_sat.groupby('Product')['Rnnmm'].mean().reset_index()
sns.barplot(data=mean_rnn, x='Product', y='Rnnmm', hue='Product', palette=prod_colors, legend=False, ax=ax, width=0.5)
for i, v in enumerate(mean_rnn['Rnnmm']):
    ax.text(i, v + 0.2, f"{v:.1f} hari", ha='center', fontweight='bold')
ax.set_title('(d) Rata-rata Frekuensi Hari Hujan Ekstrem (Rnnmm, nn=50 mm)')
ax.set_ylabel('Frekuensi (Hari/Tahun)')
ax.set_xlabel('Produk Satelit / Reanalisis')

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig8_satellite_intercomparison_radar_corr.png", dpi=300)
plt.close()

# =============================================================================
# FIG 9: CORRELATION MATRIX (TEMPERATURE VS PRECIPITATION EXTREMES)
# =============================================================================
print("Generating Fig 9: Correlation Matrix of 23 non-constant ETCCDI Indices...")
# Select meaningful non-constant indices
active_cols = [
    'TXx', 'TXn', 'TNx', 'TNn', 'TR', 'TX10p', 'TX90p', 'TN10p', 'TN90p', 'WSDI', 'CSDI', 'DTR',
    'Rx1day', 'Rx5day', 'SDII', 'PRCPTOT', 'R10mm', 'R20mm', 'Rnnmm', 'R95pTOT', 'R99pTOT', 'CDD', 'CWD'
]
corr_matrix = df_etccdi[active_cols].corr(method='spearman')

plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Koefisien Korelasi Spearman (rho)"},
            annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Matriks Korelasi Spearman 23 Indeks ETCCDI Non-Konstan (Kebumen, 2000-2025)", pad=20, fontsize=13)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig9_etccdi_correlation_matrix.png", dpi=300)
plt.close()

print(f"\n[SUCCESS] All 9 publication figures generated in: {FIGS_DIR}")
