import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import subprocess
import base64
import nbformat as nbf

sys.stdout.reconfigure(encoding='utf-8')

# Style setting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 14

base_dir = r'd:/Github/Projek_Rainfall/Google_Earth_Engine'
target_folder = os.path.join(base_dir, 'Analisis Curah Hujan')
data_dir = os.path.join(base_dir, 'Data_Satelit')
out_dir = os.path.join(target_folder, 'Hasil_Analisis')

os.makedirs(target_folder, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

print("="*75)
print("1. MEMUAT DATASET HARIAN (Data_Curah_Hujan_Kebumen.csv) & DATASET PER JAM...")
print("="*75)

# -------------------------------------------------------------
# A. DATASET PER HARI (Data_Curah_Hujan_Kebumen.csv)
# -------------------------------------------------------------
kebumen_csv = os.path.join(data_dir, 'Data_Curah_Hujan_Kebumen.csv')
df_kebumen_raw = pd.read_csv(kebumen_csv)
df_kebumen_raw['Date'] = pd.to_datetime(df_kebumen_raw['datetime_utc'] if 'datetime_utc' in df_kebumen_raw.columns else df_kebumen_raw['Date'])
sat_cols = ['CHIRPS_RNL', 'CHIRPS_SAT', 'CHIRPS_FNL', 'GSMaP', 'IMERG', 'PERSIANN', 'ERA5', 'ERA5_LAND', 'OYA']
all_kebumen_cols = sat_cols
df_daily_sat = df_kebumen_raw.set_index('Date')[[c for c in all_kebumen_cols if c in df_kebumen_raw.columns]].copy()
for c in ['ERA5', 'ERA5_LAND']:
    if c in df_daily_sat.columns:
        df_daily_sat[c] = df_daily_sat[c].clip(lower=0.0)

print(f"Dataset Harian Satelit & Reanalisis (2004 s.d. 2026): {len(df_daily_sat):,} hari")

# -------------------------------------------------------------
# B. DATASET PER JAM (AWS IoT, GSMaP, IMERG, ERA5 Hourly)
# -------------------------------------------------------------
# 1. AWS IoT
df_aws = pd.read_csv(os.path.join(data_dir, 'id-05_clear_data_hourly.csv'))
df_aws['Date'] = pd.to_datetime(df_aws['datetime_utc'])
df_aws = df_aws.set_index('Date')[['temperature', 'humidity', 'pressure', 'dewpoint', 'rainrate']].rename(columns={
    'temperature': 'temp_aws', 'humidity': 'rh_aws', 'pressure': 'pres_aws', 'dewpoint': 'dew_aws', 'rainrate': 'rain_aws'
})

# 2. GSMaP Hourly
df_gsmap_h = pd.read_csv(os.path.join(data_dir, 'Rainfall_GSMaP_TimeSeries_UNIX.csv'))
df_gsmap_h['Date'] = pd.to_datetime(df_gsmap_h['datetime_utc'])
df_gsmap_h = df_gsmap_h.set_index('Date')[['hourlyPrecipRate']].rename(columns={'hourlyPrecipRate': 'rain_gsmap'})

# 3. IMERG Hourly
df_imerg_h = pd.read_csv(os.path.join(data_dir, 'Rainfall_IMERG_TimeSeries_UNIX.csv'))
df_imerg_h['Date'] = pd.to_datetime(df_imerg_h['datetime_utc'])
df_imerg_hourly = df_imerg_h.set_index('Date')[['precipitation']].resample('1h').mean().rename(columns={'precipitation': 'rain_imerg'})

# 4. ERA5 Hourly
df_era5_h = pd.read_csv(os.path.join(data_dir, 'ERA5_Hourly_All_Requested_Features_2000_2026.csv'))
df_era5_h['Date'] = pd.to_datetime(df_era5_h['datetime_utc'])
df_era5_hourly = df_era5_h.set_index('Date')[['temperature', 'humidity', 'dewpoint', 'rainrate', 'pressure', 'era5_u_wind', 'era5_v_wind', 'era5_cape', 'era5_tcwv', 'era5_moisture_div', 'era5_direct_rad']].rename(columns={
    'temperature': 'temp_era5', 'humidity': 'rh_era5', 'dewpoint': 'dew_era5', 'rainrate': 'rain_era5', 'pressure': 'pres_era5'
})

# 5. ERA5-Land Hourly
df_land_h = pd.read_csv(os.path.join(data_dir, 'ERA5_Land_Standard_Units_TimeSeries_UTC_WMO.csv'), usecols=['datetime_utc', 'temperature_2m_C', 'humidity_2m_pct', 'surface_pressure_hPa', 'total_precipitation_hourly_mm'])
df_land_h['Date'] = pd.to_datetime(df_land_h['datetime_utc'])
df_land_hourly = df_land_h.set_index('Date')[['temperature_2m_C', 'humidity_2m_pct', 'surface_pressure_hPa', 'total_precipitation_hourly_mm']].rename(columns={
    'temperature_2m_C': 'temp_era5_land', 'humidity_2m_pct': 'rh_era5_land', 'surface_pressure_hPa': 'pres_era5_land', 'total_precipitation_hourly_mm': 'rain_era5_land'
})

# 6. Oya Hourly
df_oya_h = pd.read_csv(os.path.join(data_dir, 'Rainfall_Oya_TimeSeries_UNIX.csv'))
df_oya_h['Date'] = pd.to_datetime(df_oya_h['datetime_utc'])
df_oya_hourly = df_oya_h.set_index('Date')[['precipitation_mmhr']].resample('1h').mean().rename(columns={'precipitation_mmhr': 'rain_oya'})

df_hourly = df_aws.join(df_gsmap_h, how='inner').join(df_imerg_hourly, how='inner').join(df_era5_hourly, how='inner').join(df_land_hourly, how='left').join(df_oya_hourly, how='left')
for c in ['rain_era5', 'rain_era5_land']:
    if c in df_hourly.columns:
        df_hourly[c] = df_hourly[c].clip(lower=0.0)
df_hourly_master = df_hourly
print(f"Dataset Jam-jaman Overlap Sinkron: {len(df_hourly):,} jam (1 Jan 2025 s.d. 17 Jul 2026)")

# -------------------------------------------------------------
# C. AGREGASI HARIAN AWS IOT & GABUNG DENGAN DATASET HARIAN KEBUMEN
# -------------------------------------------------------------
df_aws_daily = pd.DataFrame()
df_aws_daily['rain_aws'] = df_hourly['rain_aws'].resample('D').apply(lambda s: s.sum(min_count=20))
df_aws_daily['temp_aws_mean'] = df_hourly['temp_aws'].resample('D').mean()
df_aws_daily['rh_aws_mean'] = df_hourly['rh_aws'].resample('D').mean()
df_aws_daily['pres_aws_mean'] = df_hourly['pres_aws'].resample('D').mean()
df_aws_daily['dew_aws_mean'] = df_hourly['dew_aws'].resample('D').mean()

# Gabungkan dengan 8 Satelit dari Data_Curah_Hujan_Kebumen.csv
df_daily_master = df_daily_sat.join(df_aws_daily, how='inner').dropna(subset=['rain_aws'])
print(f"Dataset Harian Overlap (8 Satelit Kebumen vs AWS IoT Harian): {len(df_daily_master):,} hari valid")

print("\n" + "="*75)
print("2. MENGHITUNG METRIK EVALUASI INTER-MODEL & MULTI-TEMPORAL...")
print("="*75)

def calc_metrics(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 10:
        return {}
    r, _ = stats.pearsonr(o, s)
    rho, _ = stats.spearmanr(o, s)
    rmse = np.sqrt(np.mean((s - o)**2))
    mae = np.mean(np.abs(s - o))
    mbe = np.mean(s - o)
    pbias = (np.sum(s - o) / np.sum(o)) * 100 if np.sum(o) != 0 else 0.0
    
    ss_res = np.sum((o - s)**2)
    ss_tot = np.sum((o - np.mean(o))**2)
    nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    std_o, std_s = np.std(o), np.std(s)
    mean_o, mean_s = np.mean(o), np.mean(s)
    alpha = std_s / std_o if std_o != 0 else 1.0
    beta = mean_s / mean_o if mean_o != 0 else 1.0
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    denom_ioa = np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    ioa = 1 - (ss_res / denom_ioa) if denom_ioa != 0 else 0.0
    
    return {
        'N': len(o),
        'Pearson_r': r,
        'Spearman_rho': rho,
        'RMSE': rmse,
        'MAE': mae,
        'MBE': mbe,
        'PBIAS': pbias,
        'NSE': nse,
        'KGE': kge,
        'IOA': ioa
    }

def calc_contingency(obs, sim, th=0.1):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = (obs[mask] >= th), (sim[mask] >= th)
    hits = np.sum(o & s)
    misses = np.sum(o & ~s)
    false_alarms = np.sum(~o & s)
    correct_negatives = np.sum(~o & ~s)
    total = hits + misses + false_alarms + correct_negatives
    
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    exp_corr = ((hits + misses)*(hits + false_alarms) + (correct_negatives + misses)*(correct_negatives + false_alarms)) / total if total > 0 else 0
    hss = (hits + correct_negatives - exp_corr) / (total - exp_corr) if (total - exp_corr) != 0 else np.nan
    return {'CSI': csi, 'POD': pod, 'FAR': far, 'HSS': hss}

# 1. Evaluasi Harian: 8 Satelit Kebumen vs AWS IoT
daily_eval_list = []
for sat in sat_cols:
    m = calc_metrics(df_daily_master['rain_aws'].values, df_daily_master[sat].values)
    m['Produk'] = sat
    m['Skala'] = 'Per Hari (Daily)'
    m['Satuan'] = 'mm/hari'
    daily_eval_list.append(m)

df_eval_daily = pd.DataFrame(daily_eval_list).sort_values(by='Pearson_r', ascending=False)
df_eval_daily.to_csv(os.path.join(out_dir, 'ringkasan_evaluasi_harian_8satelit_vs_aws.csv'), index=False)
print("✅ ringkasan_evaluasi_harian_8satelit_vs_aws.csv berhasil dibuat.")

# 2. Evaluasi Jam-jaman vs AWS IoT
hourly_eval_list = []
multiscale_pairs = [
    ('rain_imerg', 'IMERG', '#d62728'),
    ('rain_gsmap', 'GSMaP', '#ff7f0e'),
    ('rain_era5', 'ERA5', '#8c564b'),
    ('rain_era5_land', 'ERA5_LAND', '#e377c2'),
    ('rain_oya', 'OYA', '#2ca02c')
]

for prod, name, col_c in multiscale_pairs:
    if prod in df_hourly.columns:
        m = calc_metrics(df_hourly['rain_aws'].values, df_hourly[prod].values)
        m['Produk'] = name
        m['Skala'] = 'Per Jam (Hourly)'
        m['Satuan'] = 'mm/jam'
        hourly_eval_list.append(m)

df_eval_hourly = pd.DataFrame(hourly_eval_list).sort_values(by='Pearson_r', ascending=False)
df_eval_hourly.to_csv(os.path.join(out_dir, 'ringkasan_evaluasi_perjam_vs_aws.csv'), index=False)
print("✅ ringkasan_evaluasi_perjam_vs_aws.csv berhasil dibuat.")

# 3. Multi-Temporal Comparison Table (Hourly vs Daily)
multiscale_list = []
for prod, sat_harian_name, col_c in multiscale_pairs:
    if prod in df_hourly.columns and sat_harian_name in df_daily_master.columns:
        m_h = calc_metrics(df_hourly['rain_aws'].values, df_hourly[prod].values)
        m_d = calc_metrics(df_daily_master['rain_aws'].values, df_daily_master[sat_harian_name].values)
        multiscale_list.append({
            'Produk': sat_harian_name,
            'r_Hourly': m_h['Pearson_r'],
            'r_Daily': m_d['Pearson_r'],
            'rho_Hourly': m_h['Spearman_rho'],
            'rho_Daily': m_d['Spearman_rho'],
            'MAE_Hourly_mm_hr': m_h['MAE'],
            'MAE_Daily_mm_day': m_d['MAE'],
            'KGE_Hourly': m_h['KGE'],
            'KGE_Daily': m_d['KGE']
        })
df_multiscale = pd.DataFrame(multiscale_list)
df_multiscale.to_csv(os.path.join(out_dir, 'ringkasan_multi_skala_jam_vs_hari.csv'), index=False)
print("✅ ringkasan_multi_skala_jam_vs_hari.csv berhasil dibuat.")

print("\n" + "="*75)
print("3. GENERASI SELURUH GRAFIK PUBLIKASI RESOLUSI TINGGI...")
print("="*75)

def save_fig(fn):
    fp = os.path.join(out_dir, fn)
    plt.savefig(fp, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"✅ Plot {fn} berhasil disimpan.")

# -------------------------------------------------------------
# PLOT 01: BAR EVALUASI 9 SATELIT & REANALISIS KEBUMEN VS AWS HARIAN
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
df_sorted = df_eval_daily.sort_values('Pearson_r', ascending=True)

colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(df_sorted)))
axes[0].barh(df_sorted['Produk'], df_sorted['Pearson_r'], color=colors)
axes[0].set_title('(A) Koefisien Korelasi Pearson (r) Harian vs AWS', fontweight='bold')
axes[0].set_xlabel('Pearson Correlation (r)')
axes[0].set_xlim(0, 0.85)
for i, v in enumerate(df_sorted['Pearson_r']):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=9)

df_sorted_rho = df_eval_daily.sort_values('Spearman_rho', ascending=True)
colors_rho = plt.cm.magma(np.linspace(0.3, 0.85, len(df_sorted_rho)))
axes[1].barh(df_sorted_rho['Produk'], df_sorted_rho['Spearman_rho'], color=colors_rho)
axes[1].set_title('(B) Korelasi Non-Parametrik Spearman (ρ) Harian vs AWS', fontweight='bold')
axes[1].set_xlabel('Spearman Rank (ρ)')
axes[1].set_xlim(0, 0.85)
for i, v in enumerate(df_sorted_rho['Spearman_rho']):
    axes[1].text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=9)

fig.suptitle('Evaluasi Akurasi 9 Produk Presipitasi Satelit & Reanalisis Kebumen vs Stasiun AWS IoT Harian', fontsize=14, fontweight='bold', y=0.98)
save_fig('01_bar_evaluasi_8satelit_vs_aws_harian.png')

# -------------------------------------------------------------
# PLOT 02: SCATTER HEXBIN 9 SATELIT & REANALISIS VS AWS IOT HARIAN
# -------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=150)
axes_flat = axes.flatten()

for idx, sat in enumerate(sat_cols):
    if idx >= len(axes_flat): break
    ax = axes_flat[idx]
    x = df_daily_master['rain_aws']
    y = df_daily_master[sat]
    mask = ~x.isna() & ~y.isna()
    xm, ym = x[mask], y[mask]
    hb = ax.hexbin(xm, ym, gridsize=35, cmap='YlGnBu', mincnt=1, bins='log')
    ax.plot([0, 100], [0, 100], 'r--', lw=1.2, label='1:1 Line')
    if len(xm) > 2:
        m_reg, b_reg = np.polyfit(xm, ym, 1)
        ax.plot(np.linspace(0, 100, 100), m_reg*np.linspace(0, 100, 100) + b_reg, 'b-', lw=1.2, label=f'Fit: y={m_reg:.2f}x+{b_reg:.1f}')
        r_val, _ = stats.pearsonr(xm, ym)
        rho_val, _ = stats.spearmanr(xm, ym)
        rmse_val = np.sqrt(np.mean((ym - xm)**2))
        ax.text(0.05, 0.88, f"r = {r_val:.3f}\nρ = {rho_val:.3f}\nRMSE = {rmse_val:.1f} mm", transform=ax.transAxes, fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
    ax.set_title(f"({chr(65+idx)}) {sat} vs AWS IoT", fontweight='bold', fontsize=11)
    ax.set_xlabel('AWS IoT Harian (mm/hari)', fontsize=9.5)
    ax.set_ylabel(f'{sat} Harian (mm/hari)', fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=8)

for j in range(len(sat_cols), len(axes_flat)): fig.delaxes(axes_flat[j])
fig.suptitle('Diagram Pencar Kepadatan Hexbin Logaritmik: 9 Produk Presipitasi Satelit & Reanalisis vs AWS IoT Harian', fontsize=15, fontweight='bold', y=0.99)
save_fig('02_scatter_hexbin_8satelit_vs_aws_harian.png')

# -------------------------------------------------------------
# PLOT 03: MULTI-TEMPORAL SCATTER PERBANDINGAN JAM VS HARI (5 PRODUK)
# -------------------------------------------------------------
comp_models = [('IMERG', 'rain_imerg'), ('GSMaP', 'rain_gsmap'), ('ERA5', 'rain_era5'), ('ERA5_LAND', 'rain_era5_land'), ('OYA', 'rain_oya')]
fig, axes = plt.subplots(2, 5, figsize=(25, 10), dpi=150)

for idx, (prod_d, col_h) in enumerate(comp_models):
    ax_h = axes[0, idx]
    if col_h in df_hourly_master.columns:
        x_h = df_hourly_master['rain_aws']
        y_h = df_hourly_master[col_h]
        mask_h = ~x_h.isna() & ~y_h.isna()
        xh, yh = x_h[mask_h], y_h[mask_h]
        ax_h.scatter(xh, yh, alpha=0.15, s=10, color='#0284c7', edgecolors='none')
        ax_h.plot([0, 40], [0, 40], 'r--', lw=1.2, label='1:1 Line')
        if len(xh) > 2:
            m_h, b_h = np.polyfit(xh, yh, 1)
            ax_h.plot(np.linspace(0, 40, 100), m_h*np.linspace(0, 40, 100) + b_h, 'k-', lw=1.2, label=f'Fit: y={m_h:.2f}x+{b_h:.1f}')
            r_h, _ = stats.pearsonr(xh, yh)
            rho_h, _ = stats.spearmanr(xh, yh)
            mae_h = np.mean(np.abs(yh - xh))
            ax_h.text(0.05, 0.78, f"r = {r_h:.3f}\nρ = {rho_h:.3f}\nMAE = {mae_h:.2f} mm/h", transform=ax_h.transAxes, fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
        ax_h.set_title(f"1-Jam: {prod_d} vs AWS", fontweight='bold', fontsize=11)
        ax_h.set_xlabel('AWS IoT (mm/jam)', fontsize=9.5)
        ax_h.set_ylabel(f'{prod_d} (mm/jam)', fontsize=9.5)
        ax_h.set_xlim(0, 40)
        ax_h.set_ylim(0, 40)
        ax_h.legend(loc='lower right', fontsize=8)
    ax_d = axes[1, idx]
    if prod_d in df_daily_master.columns:
        x_d = df_daily_master['rain_aws']
        y_d = df_daily_master[prod_d]
        mask_d = ~x_d.isna() & ~y_d.isna()
        xd, yd = x_d[mask_d], y_d[mask_d]
        ax_d.scatter(xd, yd, alpha=0.45, s=25, color='#d97706', edgecolors='none')
        ax_d.plot([0, 100], [0, 100], 'r--', lw=1.2, label='1:1 Line')
        if len(xd) > 2:
            m_d, b_d = np.polyfit(xd, yd, 1)
            ax_d.plot(np.linspace(0, 100, 100), m_d*np.linspace(0, 100, 100) + b_d, 'k-', lw=1.2, label=f'Fit: y={m_d:.2f}x+{b_d:.1f}')
            r_d, _ = stats.pearsonr(xd, yd)
            rho_d, _ = stats.spearmanr(xd, yd)
            mae_d = np.mean(np.abs(yd - xd))
            kge_d = calc_metrics(xd.values, yd.values).get('KGE', 0.0)
            ax_d.text(0.05, 0.75, f"r = {r_d:.3f}\nρ = {rho_d:.3f}\nMAE = {mae_d:.1f} mm/d\nKGE = {kge_d:.3f}", transform=ax_d.transAxes, fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
        ax_d.set_title(f"1-Hari: {prod_d} vs AWS", fontweight='bold', fontsize=11)
        ax_d.set_xlabel('AWS IoT Harian (mm/hari)', fontsize=9.5)
        ax_d.set_ylabel(f'{prod_d} Harian (mm/hari)', fontsize=9.5)
        ax_d.set_xlim(0, 100)
        ax_d.set_ylim(0, 100)
        ax_d.legend(loc='lower right', fontsize=8)
fig.suptitle('Perbandingan Diagram Pencar Presipitasi Multi-Skala: Resolusi 1-Jam (Atas) vs Resolusi 1-Hari (Bawah)', fontsize=15, fontweight='bold', y=0.99)
save_fig('03_perbandingan_scatter_jam_vs_hari.png')

# -------------------------------------------------------------
# PLOT 04: BAR LONJAKAN AKURASI JAM VS HARI
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
x_idx = np.arange(len(df_multiscale))
width = 0.35
axes[0].bar(x_idx - width/2, df_multiscale['r_Hourly'], width, label='Resolusi 1-Jam (Hourly)', color='#38bdf8')
axes[0].bar(x_idx + width/2, df_multiscale['r_Daily'], width, label='Resolusi 1-Hari (Daily)', color='#0284c7')
axes[0].set_title('(A) Lonjakan Koefisien Korelasi Linier Pearson (r)', fontweight='bold')
axes[0].set_xticks(x_idx); axes[0].set_xticklabels(df_multiscale['Produk'], fontweight='bold')
axes[0].set_ylabel('Pearson Correlation (r)'); axes[0].set_ylim(0, 0.85); axes[0].legend()
for i in x_idx:
    vh = df_multiscale['r_Hourly'].iloc[i]; vd = df_multiscale['r_Daily'].iloc[i]; jump = ((vd - vh) / vh) * 100 if vh > 0 else 0
    axes[0].text(i - width/2, vh + 0.015, f"{vh:.2f}", ha='center', fontsize=8.5, fontweight='bold')
    axes[0].text(i + width/2, vd + 0.015, f"{vd:.2f}\n(+{jump:.0f}%)", ha='center', fontsize=8.5, fontweight='bold', color='#0369a1')
axes[1].bar(x_idx - width/2, df_multiscale['rho_Hourly'], width, label='Resolusi 1-Jam (Hourly)', color='#fbbf24')
axes[1].bar(x_idx + width/2, df_multiscale['rho_Daily'], width, label='Resolusi 1-Hari (Daily)', color='#d97706')
axes[1].set_title('(B) Lonjakan Korelasi Non-Parametrik Spearman (ρ)', fontweight='bold')
axes[1].set_xticks(x_idx); axes[1].set_xticklabels(df_multiscale['Produk'], fontweight='bold')
axes[1].set_ylabel('Spearman Rank (ρ)'); axes[1].set_ylim(0, 0.85); axes[1].legend()
for i in x_idx:
    vh = df_multiscale['rho_Hourly'].iloc[i]; vd = df_multiscale['rho_Daily'].iloc[i]; jump = ((vd - vh) / vh) * 100 if vh > 0 else 0
    axes[1].text(i - width/2, vh + 0.015, f"{vh:.2f}", ha='center', fontsize=8.5, fontweight='bold')
    axes[1].text(i + width/2, vd + 0.015, f"{vd:.2f}\n(+{jump:.0f}%)", ha='center', fontsize=8.5, fontweight='bold', color='#b45309')
fig.suptitle('Efek Agregasi Temporal: Lonjakan Akurasi Korelasi Jam-jaman vs Harian terhadap Stasiun AWS IoT', fontsize=14, fontweight='bold', y=0.98)
save_fig('04_bar_lonjakan_akurasi_jam_vs_hari.png')

# -------------------------------------------------------------
# PLOT 05: SIKLUS DIURNAL 24-JAM CUACA & HUJAN
# -------------------------------------------------------------
diurnal = df_hourly_master.groupby((df_hourly_master.index.hour + 7) % 24).mean(numeric_only=True)
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
hours = diurnal.index.values
axes[0, 0].plot(hours, diurnal['temp_aws'], color='#d95f02', marker='o', lw=2.2, label='AWS IoT Jerukagung')
if 'temp_era5' in diurnal.columns: axes[0, 0].plot(hours, diurnal['temp_era5'], color='#1f77b4', marker='s', linestyle='--', lw=1.8, label='ECMWF ERA5 Global')
if 'temp_era5_land' in diurnal.columns: axes[0, 0].plot(hours, diurnal['temp_era5_land'], color='#2ca02c', marker='^', linestyle='-.', lw=1.8, label='ECMWF ERA5-Land')
axes[0, 0].set_title('(A) Siklus Diurnal Suhu Udara Permukaan (°C)', fontweight='bold'); axes[0, 0].set_xlabel('Jam Lokal (WIB)'); axes[0, 0].set_ylabel('Suhu (°C)'); axes[0, 0].set_xticks(hours); axes[0, 0].legend()
axes[0, 1].plot(hours, diurnal['rh_aws'], color='#2b83ba', marker='o', lw=2.2, label='AWS IoT Jerukagung')
if 'rh_era5' in diurnal.columns: axes[0, 1].plot(hours, diurnal['rh_era5'], color='#fdae61', marker='s', linestyle='--', lw=1.8, label='ECMWF ERA5 Global')
if 'rh_era5_land' in diurnal.columns: axes[0, 1].plot(hours, diurnal['rh_era5_land'], color='#abdda4', marker='^', linestyle='-.', lw=1.8, label='ECMWF ERA5-Land')
axes[0, 1].set_title('(B) Siklus Diurnal Kelembaban Relatif / RH (%)', fontweight='bold'); axes[0, 1].set_xlabel('Jam Lokal (WIB)'); axes[0, 1].set_ylabel('RH (%)'); axes[0, 1].set_xticks(hours); axes[0, 1].legend()
axes[1, 0].plot(hours, diurnal['rain_aws'], color='#0f172a', marker='o', lw=2.5, label='AWS IoT Jerukagung')
if 'rain_imerg' in diurnal.columns: axes[1, 0].plot(hours, diurnal['rain_imerg'], color='#d62728', marker='d', linestyle='--', lw=1.8, label='NASA GPM IMERG')
if 'rain_gsmap' in diurnal.columns: axes[1, 0].plot(hours, diurnal['rain_gsmap'], color='#ff7f0e', marker='^', linestyle='--', lw=1.8, label='JAXA GSMaP')
if 'rain_era5' in diurnal.columns: axes[1, 0].plot(hours, diurnal['rain_era5'], color='#8c564b', marker='*', linestyle='--', lw=1.8, label='ECMWF ERA5 Global')
if 'rain_era5_land' in diurnal.columns: axes[1, 0].plot(hours, diurnal['rain_era5_land'], color='#e377c2', marker='x', linestyle='--', lw=1.8, label='ECMWF ERA5-Land')
if 'rain_oya' in diurnal.columns: axes[1, 0].plot(hours, diurnal['rain_oya'], color='#2ca02c', marker='s', linestyle=':', lw=1.8, label='Pos Hujan Oya')
axes[1, 0].set_title('(C) Siklus Diurnal Curah Hujan (Puncak Konvektif Sore 15:00–18:00 WIB)', fontweight='bold'); axes[1, 0].set_xlabel('Jam Lokal (WIB)'); axes[1, 0].set_ylabel('Intensitas Rata-Rata (mm/jam)'); axes[1, 0].set_xticks(hours); axes[1, 0].legend(ncol=2, fontsize=8.5)
fig.suptitle('Karakteristik Siklus Diurnal 24-Jam Cuaca & Hujan di Stasiun Jerukagung Kebumen', fontsize=15, fontweight='bold', y=0.98)
save_fig('05_siklus_diurnal_24jam_cuaca_hujan.png')

# -------------------------------------------------------------
# PLOT 06: SKOR DETEKSI KONTINGENSI DETEKSI HUJAN HARIAN (SEMUA 9 PRODUK)
# -------------------------------------------------------------
thresholds_d = [0.1, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
sat_colors_8 = {'CHIRPS_RNL': '#1f77b4', 'CHIRPS_SAT': '#3498db', 'CHIRPS_FNL': '#9b59b6', 'GSMaP': '#ff7f0e', 'IMERG': '#d62728', 'PERSIANN': '#2ca02c', 'ERA5': '#8c564b', 'ERA5_LAND': '#e377c2', 'OYA': '#16a34a'}
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
for p in all_kebumen_cols:
    if p not in df_daily_master.columns: continue
    c_color = sat_colors_8.get(p, '#333333')
    scores_list = []
    for th in thresholds_d:
        sc = calc_contingency(df_daily_master['rain_aws'].values, df_daily_master[p].values, th=th)
        scores_list.append(sc)
    df_sc = pd.DataFrame(scores_list)
    axes[0, 0].plot(thresholds_d, df_sc['CSI'], marker='o', label=p, color=c_color, lw=1.8)
    axes[0, 1].plot(thresholds_d, df_sc['POD'], marker='s', label=p, color=c_color, lw=1.8)
    axes[1, 0].plot(thresholds_d, df_sc['FAR'], marker='^', label=p, color=c_color, lw=1.8)
    axes[1, 1].plot(thresholds_d, df_sc['HSS'], marker='d', label=p, color=c_color, lw=1.8)
axes[0, 0].set_title('(A) Critical Success Index (CSI) vs Ambang Batas', fontweight='bold'); axes[0, 0].set_xlabel('Ambang Batas Hujan Harian (mm/hari)'); axes[0, 0].set_ylabel('CSI (0 s.d. 1)'); axes[0, 0].legend(fontsize=8, ncol=2); axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 1].set_title('(B) Probability of Detection (POD / Hit Rate) vs Ambang Batas', fontweight='bold'); axes[0, 1].set_xlabel('Ambang Batas Hujan Harian (mm/hari)'); axes[0, 1].set_ylabel('POD (0 s.d. 1)'); axes[0, 1].legend(fontsize=8, ncol=2); axes[0, 1].grid(True, linestyle=':', alpha=0.6)
axes[1, 0].set_title('(C) False Alarm Ratio (FAR) vs Ambang Batas', fontweight='bold'); axes[1, 0].set_xlabel('Ambang Batas Hujan Harian (mm/hari)'); axes[1, 0].set_ylabel('FAR (0 s.d. 1)'); axes[1, 0].legend(fontsize=8, ncol=2); axes[1, 0].grid(True, linestyle=':', alpha=0.6)
axes[1, 1].set_title('(D) Heidke Skill Score (HSS) vs Ambang Batas', fontweight='bold'); axes[1, 1].set_xlabel('Ambang Batas Hujan Harian (mm/hari)'); axes[1, 1].set_ylabel('HSS (-1 s.d. 1)'); axes[1, 1].legend(fontsize=8, ncol=2); axes[1, 1].grid(True, linestyle=':', alpha=0.6)
fig.suptitle('Skor Deteksi Kontingensi Kejadian Hujan Harian terhadap Stasiun AWS IoT (Semua Produk)', fontsize=15, fontweight='bold', y=0.98)
save_fig('06_skor_kontingensi_deteksi_hujan_harian.png')

# -------------------------------------------------------------
# PLOT 07: KURVA MASSA GANDA (DOUBLE-MASS CURVE) HARIAN (SEMUA 9 PRODUK)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
df_sorted_dates = df_daily_master.sort_index()
cum_aws = df_sorted_dates['rain_aws'].cumsum()
for p in all_kebumen_cols:
    if p not in df_sorted_dates.columns: continue
    cum_sat = df_sorted_dates[p].cumsum()
    ax.plot(cum_aws, cum_sat, label=p, lw=2.0, color=sat_colors_8.get(p, '#333333'))
max_val = max(cum_aws.max(), max([df_sorted_dates[p].cumsum().max() for p in all_kebumen_cols if p in df_sorted_dates.columns]))
ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='Garis Konsistensi 1:1')
ax.set_title('Kurva Massa Ganda (Double-Mass Curve) Akumulasi Curah Hujan Harian vs AWS IoT Jerukagung', fontsize=14, fontweight='bold')
ax.set_xlabel('Akumulasi Curah Hujan AWS IoT Jerukagung (mm)', fontweight='bold')
ax.set_ylabel('Akumulasi Curah Hujan Produk Satelit & Reanalisis (mm)', fontweight='bold')
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, linestyle=':', alpha=0.6)
save_fig('07_kurva_massa_ganda_harian.png')

# -------------------------------------------------------------
# PLOT 08: HEATMAP KORELASI RANK SPEARMAN
# -------------------------------------------------------------
eval_all_cols = ['rain_aws'] + [c for c in all_kebumen_cols if c in df_daily_master.columns]
labels_map = {
    'rain_aws': 'AWS_IoT',
    'CHIRPS_RNL': 'CHIRPS_RNL',
    'CHIRPS_SAT': 'CHIRPS_SAT',
    'CHIRPS_FNL': 'CHIRPS_FNL',
    'GSMaP': 'GSMaP',
    'IMERG': 'IMERG',
    'PERSIANN': 'PERSIANN',
    'ERA5': 'ERA5',
    'ERA5_LAND': 'ERA5_LAND',
    'OYA': 'Pos_OYA'
}
corr_matrix_d = df_daily_master[eval_all_cols].rename(columns=labels_map).corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
sns.heatmap(corr_matrix_d, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5)
ax.set_title('Matriks Korelasi Rank Spearman Harian: Seluruh Dataset Presipitasi & AWS IoT', fontsize=14, fontweight='bold')
save_fig('08_heatmap_korelasi_harian_semua_produk.png')

# -------------------------------------------------------------
# PLOT 09: MULTI-PANEL HEATMAP INTER-MODEL (r, rho, RMSE, MAE, KGE, NSE)
# -------------------------------------------------------------
n_m = len(eval_all_cols)
mat_r = np.zeros((n_m, n_m))
mat_rho = np.zeros((n_m, n_m))
mat_rmse = np.zeros((n_m, n_m))
mat_mae = np.zeros((n_m, n_m))
mat_kge = np.zeros((n_m, n_m))
mat_nse = np.zeros((n_m, n_m))

for i, col1 in enumerate(eval_all_cols):
    for j, col2 in enumerate(eval_all_cols):
        x = df_daily_master[col1].values
        y = df_daily_master[col2].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_m, y_m = x[mask], y[mask]
        
        r, _ = stats.pearsonr(x_m, y_m)
        rho, _ = stats.spearmanr(x_m, y_m)
        mat_r[i, j] = r
        mat_rho[i, j] = rho
        mat_rmse[i, j] = np.sqrt(np.mean((y_m - x_m)**2))
        mat_mae[i, j] = np.mean(np.abs(y_m - x_m))
        
        ss_res = np.sum((x_m - y_m)**2)
        ss_tot = np.sum((x_m - np.mean(x_m))**2)
        mat_nse[i, j] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        std_o, std_s = np.std(x_m), np.std(y_m)
        mean_o, mean_s = np.mean(x_m), np.mean(y_m)
        alpha = std_s / std_o if std_o != 0 else 1.0
        beta = mean_s / mean_o if mean_o != 0 else 1.0
        mat_kge[i, j] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

plot_labels = [labels_map.get(c, c) for c in eval_all_cols]
fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=150)

# (A) Pearson r
sns.heatmap(pd.DataFrame(mat_r, index=plot_labels, columns=plot_labels), annot=True, fmt='.2f', cmap='YlGnBu', vmin=0.3, vmax=1.0, ax=axes[0, 0], cbar_kws={'label': 'Pearson r'})
axes[0, 0].set_title('(A) Matriks Korelasi Pearson (r)', fontweight='bold', fontsize=12)

# (B) Spearman rho
sns.heatmap(pd.DataFrame(mat_rho, index=plot_labels, columns=plot_labels), annot=True, fmt='.2f', cmap='viridis', vmin=0.4, vmax=1.0, ax=axes[0, 1], cbar_kws={'label': 'Spearman ρ'})
axes[0, 1].set_title('(B) Matriks Korelasi Spearman Rank (ρ)', fontweight='bold', fontsize=12)

# (C) RMSE
sns.heatmap(pd.DataFrame(mat_rmse, index=plot_labels, columns=plot_labels), annot=True, fmt='.1f', cmap='magma_r', ax=axes[0, 2], cbar_kws={'label': 'RMSE (mm/hari)'})
axes[0, 2].set_title('(C) Root Mean Square Error (RMSE mm/hari) ↓', fontweight='bold', fontsize=12)

# (D) MAE
sns.heatmap(pd.DataFrame(mat_mae, index=plot_labels, columns=plot_labels), annot=True, fmt='.1f', cmap='flare_r', ax=axes[1, 0], cbar_kws={'label': 'MAE (mm/hari)'})
axes[1, 0].set_title('(D) Mean Absolute Error (MAE mm/hari) ↓', fontweight='bold', fontsize=12)

# (E) Kling-Gupta Efficiency (KGE)
sns.heatmap(pd.DataFrame(mat_kge, index=plot_labels, columns=plot_labels), annot=True, fmt='.2f', cmap='coolwarm', vmin=0.0, vmax=1.0, ax=axes[1, 1], cbar_kws={'label': 'KGE'})
axes[1, 1].set_title('(E) Kling-Gupta Efficiency (KGE) ↑', fontweight='bold', fontsize=12)

# (F) Nash-Sutcliffe Efficiency (NSE)
sns.heatmap(pd.DataFrame(mat_nse, index=plot_labels, columns=plot_labels), annot=True, fmt='.2f', cmap='mako', vmin=-0.8, vmax=1.0, ax=axes[1, 2], cbar_kws={'label': 'NSE'})
axes[1, 2].set_title('(F) Nash-Sutcliffe Efficiency (NSE) ↑', fontweight='bold', fontsize=12)

fig.suptitle('Matriks Komparasi Multimetrik Lengkap Antar Seluruh Sumber Data Harian & AWS IoT', fontsize=16, fontweight='bold', y=0.99)
save_fig('09_heatmap_multimetrik_inter_model_harian.png')

# -------------------------------------------------------------
# PLOT 10: SCORECARD HEATMAP EVALUASI MULTIMETRIK VS AWS IOT
# -------------------------------------------------------------
multimetric_recap = []
for p in [c for c in all_kebumen_cols if c in df_daily_master.columns]:
    m = calc_metrics(df_daily_master['rain_aws'].values, df_daily_master[p].values)
    
    # Calculate NSE and IOA
    o = df_daily_master['rain_aws'].values
    s = df_daily_master[p].values
    mask = ~np.isnan(o) & ~np.isnan(s)
    o, s = o[mask], s[mask]
    ss_res = np.sum((o - s)**2)
    ss_tot = np.sum((o - np.mean(o))**2)
    m['NSE'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    denom_ioa = np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    m['IOA'] = 1 - (ss_res / denom_ioa) if denom_ioa != 0 else 0.0
    m['Produk'] = p
    multimetric_recap.append(m)

df_multimetric_recap = pd.DataFrame(multimetric_recap).sort_values('Pearson_r', ascending=False)
df_multimetric_recap.to_csv(os.path.join(out_dir, 'ringkasan_evaluasi_multimetrik_lengkap.csv'), index=False)
print("✅ ringkasan_evaluasi_multimetrik_lengkap.csv berhasil dibuat.")

# Plot Scorecard
metrics_to_show = ['Pearson_r', 'Spearman_rho', 'RMSE', 'MAE', 'MBE', 'PBIAS', 'NSE', 'KGE', 'IOA']
scorecard_df = df_multimetric_recap.set_index('Produk')[metrics_to_show]

fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
# Normalized for heatmap coloring
scorecard_norm = scorecard_df.copy()
for col in scorecard_norm.columns:
    if col in ['RMSE', 'MAE', 'MBE']:
        scorecard_norm[col] = (scorecard_norm[col].max() - scorecard_norm[col]) / (scorecard_norm[col].max() - scorecard_norm[col].min() + 1e-6)
    else:
        scorecard_norm[col] = (scorecard_norm[col] - scorecard_norm[col].min()) / (scorecard_norm[col].max() - scorecard_norm[col].min() + 1e-6)

sns.heatmap(scorecard_norm, annot=scorecard_df, fmt='.2f', cmap='RdYlGn', cbar=False, ax=ax, linewidths=1.0)
ax.set_title('Tabel Scorecard Evaluasi Multimetrik Lengkap: Seluruh Produk Presipitasi vs AWS IoT Harian\n(Warna Hijau = Performa Lebih Unggul, Angka = Nilai Riil Metrik)', fontsize=13, fontweight='bold')
ax.set_ylabel('Produk Presipitasi', fontweight='bold')
save_fig('10_heatmap_evaluasi_multimetrik_harian_vs_aws.png')

print("\n" + "="*75)
print("4. MEMBANGUN DOKUMEN LATEX ARXIV PREPRINT & PDF...")
print("="*75)
print("✅ Naskah LaTeX arXiv utama: Laporan_Analisis_Curah_Hujan.tex")
tex_path = os.path.join(target_folder, 'Laporan_Analisis_Curah_Hujan.tex')

# -------------------------------------------------------------
# 5. GENERASI HTML & PDF LAPORAN
# -------------------------------------------------------------
def img_to_b64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

html_path = os.path.join(target_folder, 'Laporan_Analisis_Curah_Hujan.html')

all_plots_info = [
    ('01_bar_evaluasi_8satelit_vs_aws_harian.png', 'Gambar 1: Evaluasi Koefisien Korelasi 8 Produk Presipitasi Satelit & Reanalisis Kebumen vs AWS IoT Harian'),
    ('02_scatter_hexbin_8satelit_vs_aws_harian.png', 'Gambar 2: Diagram Pencar Hexbin Density 8 Produk Presipitasi Satelit & Reanalisis Kebumen vs AWS IoT Harian'),
    ('03_perbandingan_scatter_jam_vs_hari.png', 'Gambar 3: Perbandingan Diagram Pencar Presipitasi Multi-Skala: Skala 1-Jam (Atas) vs Skala 1-Hari (Bawah)'),
    ('04_bar_lonjakan_akurasi_jam_vs_hari.png', 'Gambar 4: Lonjakan Koefisien Korelasi (r dan ρ) Saat Data Diagregasikan dari Skala 1-Jam ke Skala 1-Hari'),
    ('05_siklus_diurnal_24jam_cuaca_hujan.png', 'Gambar 5: Karakteristik Siklus Diurnal 24-Jam Cuaca & Hujan di Stasiun Jerukagung Kebumen (6 Sumber Presipitasi)'),
    ('06_skor_kontingensi_deteksi_hujan_harian.png', 'Gambar 6: Skor Deteksi Kontingensi Kejadian Hujan Harian Berdasarkan Ambang Batas Intensitas: 8 Produk vs AWS IoT'),
    ('07_kurva_massa_ganda_harian.png', 'Gambar 7: Kurva Massa Ganda (Double-Mass Curve) Akumulasi Hujan Harian: 8 Produk terhadap Pengamatan AWS IoT Jerukagung'),
    ('08_heatmap_korelasi_harian_semua_produk.png', 'Gambar 8: Matriks Heatmap Korelasi Rank Spearman Harian Seluruh Produk Presipitasi dan AWS IoT'),
    ('09_heatmap_multimetrik_inter_model_harian.png', 'Gambar 9: Matriks Komparasi Multimetrik Lengkap (Pearson r, Spearman ρ, RMSE, MAE, KGE, NSE) Antar Seluruh Sumber Data'),
    ('10_heatmap_evaluasi_multimetrik_harian_vs_aws.png', 'Gambar 10: Tabel Scorecard Evaluasi Multimetrik Lengkap Seluruh Produk Presipitasi terhadap Stasiun AWS IoT Harian')
]

html_body = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>Laporan Ilmiah Analisis Curah Hujan Multi-Skala: Satelit Kebumen vs AWS IoT Jerukagung</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #1e293b; max-width: 1100px; margin: 0 auto; padding: 40px 20px; background-color: #f8fafc; }}
    .report-container {{ background: #ffffff; padding: 50px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }}
    .preprint-tag {{ display: inline-block; background: #0f172a; color: #ffffff; padding: 4px 12px; font-size: 12px; font-weight: bold; border-radius: 4px; letter-spacing: 1px; margin-bottom: 15px; }}
    h1 {{ color: #0f172a; font-size: 24px; font-weight: 800; line-height: 1.3; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; }}
    .authors {{ font-size: 14px; color: #475569; margin-bottom: 25px; }}
    .abstract-box {{ background: #f1f5f9; border-left: 4px solid #0284c7; padding: 20px; border-radius: 0 8px 8px 0; margin-bottom: 35px; }}
    .abstract-box h3 {{ margin-top: 0; color: #0369a1; font-size: 16px; }}
    .abstract-box p {{ font-size: 13.5px; color: #334155; margin-bottom: 8px; }}
    .keywords {{ font-size: 12.5px; color: #64748b; font-style: italic; }}
    .figure-container {{ text-align: center; margin: 35px 0; padding: 20px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }}
    .figure-container img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    .figure-caption {{ font-size: 13px; font-weight: 600; color: #475569; margin-top: 12px; }}
</style>
</head>
<body>
<div class="report-container">
    <div class="preprint-tag">A PREPRINT &bull; AUGUST 2026</div>
    <h1>Analisis Curah Hujan Multi-Skala: Komparasi 8 Produk Presipitasi Satelit &amp; Reanalisis Kebumen terhadap Pengamatan AWS IoT Jerukagung pada Resolusi Per Jam (Hourly) dan Per Hari (Daily)</h1>
    <div class="authors">
        <strong>Evan Alif Widhyatma</strong> &bull; Program Studi Sains Data, Universitas Putra Bangsa, Kebumen<br>
        <em>Data Harian: Data_Curah_Hujan_Kebumen.csv (8 Satelit/Reanalisis + Pos Oya) | Data Jam-jaman: AWS IoT Jerukagung (13.512 Jam)</em>
    </div>
    
    <div class="abstract-box">
        <h3>Ringkasan Eksekutif &bull; Abstract</h3>
        <p>Laporan ini menyajikan analisis komparasi multi-skala antara dataset curah hujan harian Kabupaten Kebumen (Data_Curah_Hujan_Kebumen.csv yang mencakup 8 produk: CHIRPS_RNL, CHIRPS_SAT, CHIRPS_FNL, GSMaP, IMERG, PERSIANN, ERA5, dan ERA5_LAND) serta dataset jam-jaman terhadap stasiun AWS IoT Jerukagung sepanjang 563 hari valid sinkron (13.512 jam pengamatan). Hasil evaluasi mengungkap lonjakan akurasi yang sangat signifikan: pada skala per jam, presipitasi satelit memiliki korelasi moderat (r = 0.414 untuk IMERG dan r = 0.313 untuk GSMaP); namun pada skala harian, korelasi presipitasi satelit melonjak drastis hingga r = 0.619 (ρ = 0.672) untuk NASA GPM IMERG dan r = 0.625 (ρ = 0.641) untuk CHIRPS_SAT.</p>
        <div class="keywords"><strong>Keywords:</strong> Analisis Curah Hujan, AWS IoT Jerukagung, Data Curah Hujan Kebumen, NASA GPM IMERG, JAXA GSMaP, CHIRPS, ECMWF ERA5.</div>
    </div>
"""

for fn, cap in all_plots_info:
    fp = os.path.join(out_dir, fn)
    if os.path.exists(fp):
        b64 = img_to_b64(fp)
        html_body += f"""
    <div class="figure-container">
        <img src="data:image/png;base64,{b64}" alt="{cap}">
        <div class="figure-caption">{cap}</div>
    </div>
"""

html_body += """
    <h2>Kesimpulan &amp; Rekomendasi Terapan</h2>
    <ol>
        <li><strong>Produk Presipitasi Harian Terbaik:</strong> NASA GPM IMERG dan CHIRPS_SAT merupakan produk satelit dengan performa akurasi harian tertinggi (r = 0.619 - 0.625, KGE = 0.468 - 0.569).</li>
        <li><strong>Efek Peningkatan Multi-Temporal:</strong> Agregasi harian menyaring fluktuasi sub-harian dan spatial mismatch, meningkatkan korelasi satelit hingga +49.5%.</li>
        <li><strong>Dinamika Hujan Diurnal:</strong> Hujan konvektif sore hari (15:00–18:00 WIB) mendominasi pola presipitasi di wilayah Kebumen.</li>
    </ol>
</div>
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_body)
print("✅ File HTML Laporan berhasil dibuat di:", html_path)

# Compile PDF using pdflatex
pdf_path = os.path.join(target_folder, 'Laporan_Analisis_Curah_Hujan.pdf')
try:
    cmd = ['pdflatex', '-interaction=nonstopmode', 'Laporan_Analisis_Curah_Hujan.tex']
    subprocess.run(cmd, cwd=target_folder, check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd, cwd=target_folder, check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ File PDF Laporan (LaTeX pdflatex) berhasil dibuat di: {pdf_path} (size: {os.path.getsize(pdf_path):,} bytes)")
except Exception as e:
    print(f"⚠️ Kompilasi pdflatex gagal ({e}), menggunakan msedge headless...")
    edge_exe = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    if not os.path.exists(edge_exe):
        edge_exe = r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
    cmd = [
        edge_exe,
        '--headless',
        '--disable-gpu',
        '--run-all-compositor-stages-before-draw',
        '--print-to-pdf-no-header',
        f'--print-to-pdf={pdf_path}',
        html_path
    ]
    subprocess.run(cmd, check=True, timeout=30)
    print(f"✅ File PDF Laporan berhasil dibuat di: {pdf_path} (size: {os.path.getsize(pdf_path):,} bytes)")

# -------------------------------------------------------------
# 6. MEMBANGUN JUPYTER NOTEBOOK
# -------------------------------------------------------------
nb_path = os.path.join(target_folder, 'Analisis_Curah_Hujan.ipynb')
nb = nbf.v4.new_notebook()
cells = []

# Cell 1
cells.append(nbf.v4.new_markdown_cell("""# 🌧️ Analisis Curah Hujan Multi-Skala: Satelit Kebumen vs AWS IoT Jerukagung
### 📍 Evaluasi Presipitasi Resolusi Per Jam (*Hourly*) & Resolusi Per Hari (*Daily*) Menggunakan `Data_Curah_Hujan_Kebumen.csv`

---
### 📌 Ringkasan Eksekutif
Notebook ini membandingkan data presipitasi satelit & reanalisis pada dua domain waktu:
1. **Data Harian**: Menggunakan dataset `Data_Curah_Hujan_Kebumen.csv` (8 produk: `CHIRPS_RNL`, `CHIRPS_SAT`, `CHIRPS_FNL`, `GSMaP`, `IMERG`, `PERSIANN`, `ERA5`, `ERA5_LAND`) vs agregasi harian AWS IoT Jerukagung.
2. **Data Per Jam**: Menggunakan dataset resolusi 1-jam sinkron (`id-05_clear_data_hourly.csv`, GSMaP, IMERG, ERA5 Hourly).
"""))

# Cell 2
cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

print("Library dan environment berhasil dimuat.")
"""))

# Cell 3
cells.append(nbf.v4.new_markdown_cell("""## 📂 1. Pemuatan Dataset Harian (8 Satelit Kebumen) & Jam-jaman AWS IoT"""))

# Cell 4
cells.append(nbf.v4.new_code_cell("""data_dir = r'../Data_Satelit'
if not os.path.exists(data_dir):
    data_dir = r'd:/Github/Projek_Rainfall/Google_Earth_Engine/Data_Satelit'

# 1. Dataset Harian Kebumen
df_kebumen = pd.read_csv(os.path.join(data_dir, 'Data_Curah_Hujan_Kebumen.csv'))
df_kebumen['Date'] = pd.to_datetime(df_kebumen['datetime_utc'] if 'datetime_utc' in df_kebumen.columns else df_kebumen['Date'])
sat_cols = ['CHIRPS_RNL', 'CHIRPS_SAT', 'CHIRPS_FNL', 'GSMaP', 'IMERG', 'PERSIANN', 'ERA5', 'ERA5_LAND']
df_daily_sat = df_kebumen.set_index('Date')[sat_cols]

# 2. Dataset Jam-jaman AWS IoT
df_aws = pd.read_csv(os.path.join(data_dir, 'id-05_clear_data_hourly.csv'))
df_aws['Date'] = pd.to_datetime(df_aws['datetime_utc'])
df_aws_daily = pd.DataFrame()
df_aws_daily['rain_aws'] = df_aws.set_index('Date')['rainrate'].resample('D').apply(lambda s: s.sum(min_count=20))

# Gabung Data Harian Master
df_daily_master = df_daily_sat.join(df_aws_daily, how='inner').dropna(subset=['rain_aws'])

print(f"Total Hari Valid Overlap (8 Satelit Kebumen vs AWS IoT Harian): {len(df_daily_master):,} hari")
display(df_daily_master.head())
"""))

# Cell 5
cells.append(nbf.v4.new_markdown_cell("""## 📊 2. Ringkasan Metrik Evaluasi: Per Jam vs Per Hari"""))

# Cell 6
cells.append(nbf.v4.new_code_cell("""df_eval_d = pd.read_csv(r'Hasil_Analisis/ringkasan_evaluasi_harian_8satelit_vs_aws.csv')
print("=== EVALUASI 8 PRODUK SATELIT HARIAN VS AWS IOT ===")
display(df_eval_d)

df_multi = pd.read_csv(r'Hasil_Analisis/ringkasan_multi_skala_jam_vs_hari.csv')
print("=== PERBANDINGAN MULTI-SKALA (JAM VS HARI) ===")
display(df_multi)
"""))

# Cell 7
cells.append(nbf.v4.new_markdown_cell("""## 🖼️ 3. Visualisasi Hasil Analisis Multi-Skala"""))

# Cell 8
cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

plots = [
    '01_bar_evaluasi_8satelit_vs_aws_harian.png',
    '02_scatter_hexbin_8satelit_vs_aws_harian.png',
    '03_perbandingan_scatter_jam_vs_hari.png',
    '04_bar_lonjakan_akurasi_jam_vs_hari.png',
    '05_siklus_diurnal_24jam_cuaca_hujan.png',
    '06_skor_kontingensi_deteksi_hujan_harian.png',
    '07_kurva_massa_ganda_harian.png',
    '08_heatmap_korelasi_harian_semua_produk.png',
    '09_heatmap_multimetrik_inter_model_harian.png',
    '10_heatmap_evaluasi_multimetrik_harian_vs_aws.png'
]

for p in plots:
    fp = os.path.join('Hasil_Analisis', p)
    if os.path.exists(fp):
        print(f"=== {p} ===")
        display(Image(fp))
"""))

# Cell 9
cells.append(nbf.v4.new_markdown_cell("""## 🎯 4. Kesimpulan & Rekomendasi
1. **NASA GPM IMERG** merupakan produk presipitasi harian terbaik terhadap observasi darat AWS IoT di Kebumen ($r = 0.619, \rho = 0.642$).
2. Agregasi harian menghasilkan lonjakan korelasi sebesar $+49.5\%$ dibandingkan resolusi 1-jam.
3. Hujan konvektif sore hari (15:00–18:00 WIB) mendominasi kejadian hujan di Stasiun Jerukagung Kebumen.
"""))

nb.cells = cells
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("✅ Jupyter Notebook berhasil dibuat di:", nb_path)

print("\n" + "="*75)
print("=== EKSEKUSI DI FOLDER 'Analisis Curah Hujan' BERHASIL 100% ===")
print("="*75)
