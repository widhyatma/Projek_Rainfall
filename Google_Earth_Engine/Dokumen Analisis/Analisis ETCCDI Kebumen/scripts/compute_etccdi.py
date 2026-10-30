"""
Script: compute_etccdi.py
Menghitung 27 Indeks Iklim Ekstrem ETCCDI (16 Suhu + 11 Presipitasi)
Periode Analisis: 2000 - 2025 (26 Tahun Penuh)
Baseline Period: 2001 - 2020 (Standar WMO 20 Tahun)
Lokasi: Kabupaten Kebumen (Studi Kasus: Stasiun AWS Jerukagung, Klirong)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats

# Paths
BASE_DIR = Path(r"d:\Github\Projek_Rainfall")
DATA_DIR = BASE_DIR / "Google_Earth_Engine" / "Data_Satelit"
OUT_DIR = BASE_DIR / "Google_Earth_Engine" / "Dokumen Analisis" / "Analisis ETCCDI Kebumen"
TABLES_DIR = OUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD & PREPARE DAILY METEOROLOGICAL TIME-SERIES
# -----------------------------------------------------------------------------
def load_and_aggregate_daily_era5():
    era5_file = DATA_DIR / "ERA5_Land_Standard_Units_TimeSeries_UTC_WMO.csv"
    print(f"Loading ERA5-Land hourly data from: {era5_file.name} ...")
    
    df = pd.read_csv(era5_file, usecols=['datetime_utc', 'temperature_2m_C', 'total_precipitation_hourly_mm'])
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Konversi ke Waktu Indonesia Barat (WIB, UTC+7)
    df['local_time'] = df['datetime_utc'] + pd.Timedelta(hours=7)
    df['date'] = df['local_time'].dt.date
    
    # Agregasi harian
    daily = df.groupby('date').agg(
        TX=('temperature_2m_C', 'max'),
        TN=('temperature_2m_C', 'min'),
        TG=('temperature_2m_C', 'mean'),
        PR=('total_precipitation_hourly_mm', 'sum')
    ).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    daily['day'] = daily['date'].dt.day
    daily['dayofyear'] = daily['date'].dt.dayofyear
    
    # Filter periode penuh 2000 - 2025 (26 tahun)
    daily = daily[(daily['year'] >= 2000) & (daily['year'] <= 2025)].copy()
    daily.sort_values('date', inplace=True)
    daily.reset_index(drop=True, inplace=True)
    
    print(f"Daily ERA5-Land series prepared: {len(daily):,} days ({daily['year'].min()} - {daily['year'].max()})")
    return daily

# -----------------------------------------------------------------------------
# 2. LOAD MULTI-SATELLITE PRECIPITATION BENCHMARKS
# -----------------------------------------------------------------------------
def load_satellite_precipitation():
    sat_file = DATA_DIR / "Data_Curah_Hujan_Kebumen.csv"
    print(f"Loading Multi-Satellite daily precipitation from: {sat_file.name} ...")
    
    df_sat = pd.read_csv(sat_file)
    df_sat['date'] = pd.to_datetime(df_sat['datetime_utc']).dt.date
    df_sat['date'] = pd.to_datetime(df_sat['date'])
    df_sat['year'] = df_sat['date'].dt.year
    
    # Filter 2000 - 2025
    df_sat = df_sat[(df_sat['year'] >= 2000) & (df_sat['year'] <= 2025)].copy()
    df_sat.sort_values('date', inplace=True)
    df_sat.reset_index(drop=True, inplace=True)
    
    print(f"Daily Satellite series prepared: {len(df_sat):,} days across products: CHIRPS_RNL, GSMaP, IMERG, ERA5, ERA5_LAND")
    return df_sat

# -----------------------------------------------------------------------------
# 3. ETCCDI CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_consecutive_spells(condition_series, min_length=6):
    """Menghitung total hari dalam periode beruntun minimal min_length hari."""
    in_spell = False
    cur_len = 0
    total_days = 0
    
    # We can detect groups of consecutive True
    groups = (~condition_series).cumsum()[condition_series]
    counts = condition_series.groupby(groups).transform('count')
    # Count days that belong to a run of at least min_length
    qualifying_days = condition_series & (counts >= min_length)
    return qualifying_days.sum()

def calculate_max_consecutive(condition_series):
    """Menghitung durasi beruntun maksimum (misal CDD atau CWD)."""
    max_c = 0
    cur_c = 0
    for val in condition_series:
        if val:
            cur_c += 1
            if cur_c > max_c:
                max_c = cur_c
        else:
            cur_c = 0
    return max_c

def calculate_growing_season_length(year_df):
    """Calculate ETCCDI GSL from six-day warm and cold spells."""
    warm = year_df['TG'].to_numpy() > 5.0
    cold = year_df['TG'].to_numpy() < 5.0

    def first_spell(condition, start=0):
        for index in range(start, len(condition) - 5):
            if condition[index:index + 6].all():
                return index
        return None

    start = first_spell(warm)
    if start is None:
        return 0

    end = first_spell(cold, start=max(0, 181 - 5))
    if end is None:
        return len(year_df) - start
    return max(0, end - start)

def compute_all_27_etccdi(daily_df, base_start=2001, base_end=2020):
    print("\nComputing 27 ETCCDI Core Climate Indices (2000-2025)...")
    print(f"Using baseline period {base_start}-{base_end} for percentile thresholds.")
    
    # Baseline mask
    base_mask = (daily_df['year'] >= base_start) & (daily_df['year'] <= base_end)
    base_df = daily_df[base_mask].copy()
    
    # Pre-calculate calendar day percentiles for temperature using 5-day running window
    # Days 1 to 366
    cal_tx10 = {}
    cal_tx90 = {}
    cal_tn10 = {}
    cal_tn90 = {}
    
    # Use a 365-day calendar; February 29 is assigned to February 28.
    daily_df['doy365'] = daily_df['dayofyear']
    leap_day = (daily_df['date'].dt.month == 2) & (daily_df['date'].dt.day == 29)
    after_february = daily_df['date'].dt.is_leap_year & (daily_df['date'].dt.month > 2)
    daily_df.loc[leap_day, 'doy365'] = 59
    daily_df.loc[after_february, 'doy365'] -= 1
    base_df = daily_df[base_mask].copy()

    for d in range(1, 366):
        # 5-day window centered on d: [d-2, d-1, d, d+1, d+2]
        window_days = [(d + offset - 1) % 365 + 1 for offset in [-2, -1, 0, 1, 2]]
        sub_base = base_df[base_df['doy365'].isin(window_days)]
        
        cal_tx10[d] = np.percentile(sub_base['TX'], 10)
        cal_tx90[d] = np.percentile(sub_base['TX'], 90)
        cal_tn10[d] = np.percentile(sub_base['TN'], 10)
        cal_tn90[d] = np.percentile(sub_base['TN'], 90)
        
    # Map percentiles back to daily_df
    daily_df['tx10_th'] = daily_df['doy365'].map(cal_tx10)
    daily_df['tx90_th'] = daily_df['doy365'].map(cal_tx90)
    daily_df['tn10_th'] = daily_df['doy365'].map(cal_tn10)
    daily_df['tn90_th'] = daily_df['doy365'].map(cal_tn90)
    
    # Baseline wet-day precipitation percentiles (PR >= 1.0 mm)
    wet_base = base_df[base_df['PR'] >= 1.0]['PR']
    pr_p95 = np.percentile(wet_base, 95)
    pr_p99 = np.percentile(wet_base, 99)
    print(f"Precipitation Baseline Thresholds (wet days >= 1.0mm):")
    print(f"  95th percentile (Very Wet Days):      {pr_p95:.2f} mm/day")
    print(f"  99th percentile (Extremely Wet Days): {pr_p99:.2f} mm/day")
    
    # Rolling 5-day precipitation for Rx5day
    # Calculate indices by year
    years = sorted(daily_df['year'].unique())
    records = []
    
    for yr in years:
        ydf = daily_df[daily_df['year'] == yr].copy().reset_index(drop=True)
        N_days = len(ydf)
        
        # --- TEMPERATURE INDICES (16) ---
        # 1. TXx: Highest daily maximum temp
        txx = ydf['TX'].max()
        # 2. TXn: Lowest daily maximum temp
        txn = ydf['TX'].min()
        # 3. TNx: Highest daily minimum temp
        tnx = ydf['TN'].max()
        # 4. TNn: Lowest daily minimum temp
        tnn = ydf['TN'].min()
        # 5. SU: Summer days (TX > 25 C)
        su = (ydf['TX'] > 25.0).sum()
        # 6. TR: Tropical nights (TN > 20 C)
        tr = (ydf['TN'] > 20.0).sum()
        # 7. FD: Frost days (TN < 0 C)
        fd = (ydf['TN'] < 0.0).sum()
        # 8. ID: Ice days (TX < 0 C)
        id_days = (ydf['TX'] < 0.0).sum()
        # 9. TX10p: Percentage of days TX < 10th percentile
        tx10p = (ydf['TX'] < ydf['tx10_th']).sum() / N_days * 100.0
        # 10. TX90p: Percentage of days TX > 90th percentile
        tx90p = (ydf['TX'] > ydf['tx90_th']).sum() / N_days * 100.0
        # 11. TN10p: Percentage of days TN < 10th percentile
        tn10p = (ydf['TN'] < ydf['tn10_th']).sum() / N_days * 100.0
        # 12. TN90p: Percentage of days TN > 90th percentile
        tn90p = (ydf['TN'] > ydf['tn90_th']).sum() / N_days * 100.0
        # 13. WSDI: Warm spell duration index (>= 6 days TX > 90th)
        wsdi = calculate_consecutive_spells(ydf['TX'] > ydf['tx90_th'], min_length=6)
        # 14. CSDI: Cold spell duration index (>= 6 days TN < 10th)
        csdi = calculate_consecutive_spells(ydf['TN'] < ydf['tn10_th'], min_length=6)
        # 15. DTR: Daily temperature range (mean of TX - TN)
        dtr = (ydf['TX'] - ydf['TN']).mean()
        # 16. GSL: Growing season length from ETCCDI six-day spells
        gsl = calculate_growing_season_length(ydf)
        
        # --- PRECIPITATION INDICES (11) ---
        # 17. Rx1day: Maximum 1-day precipitation
        rx1day = ydf['PR'].max()
        # 18. Rx5day: Maximum consecutive 5-day precipitation
        rx5day = ydf['PR'].rolling(5, min_periods=5).sum().max()
        # Wet days (RR >= 1.0 mm)
        wet_mask = ydf['PR'] >= 1.0
        wet_days_count = wet_mask.sum()
        # 19. SDII: Simple daily intensity index (sum PR wet / count wet)
        sdii = ydf.loc[wet_mask, 'PR'].sum() / max(1, wet_days_count) if wet_days_count > 0 else 0.0
        # 20. PRCPTOT: Total precipitation in wet days (RR >= 1.0 mm)
        prcptot = ydf.loc[wet_mask, 'PR'].sum()
        # 21. R10mm: Number of days with PR >= 10 mm
        r10mm = (ydf['PR'] >= 10.0).sum()
        # 22. R20mm: Number of days with PR >= 20 mm
        r20mm = (ydf['PR'] >= 20.0).sum()
        # 23. Rnnmm: Rnnmm with nn=50 mm for the BMKG heavy-rain threshold
        rnnmm = (ydf['PR'] >= 50.0).sum()
        # 24. R95pTOT: Precipitation from very wet days (PR > 95th percentile of baseline wet days)
        r95ptot = ydf.loc[wet_mask & (ydf['PR'] > pr_p95), 'PR'].sum()
        # 25. R99pTOT: Precipitation from extremely wet days (PR > 99th percentile of baseline wet days)
        r99ptot = ydf.loc[wet_mask & (ydf['PR'] > pr_p99), 'PR'].sum()
        # 26. CDD: Consecutive dry days (PR < 1.0 mm)
        cdd = calculate_max_consecutive(ydf['PR'] < 1.0)
        # 27. CWD: Consecutive wet days (PR >= 1.0 mm)
        cwd = calculate_max_consecutive(ydf['PR'] >= 1.0)
        
        records.append({
            'year': yr,
            # Temperature (16)
            'TXx': txx,
            'TXn': txn,
            'TNx': tnx,
            'TNn': tnn,
            'SU': su,
            'TR': tr,
            'FD': fd,
            'ID': id_days,
            'TX10p': tx10p,
            'TX90p': tx90p,
            'TN10p': tn10p,
            'TN90p': tn90p,
            'WSDI': wsdi,
            'CSDI': csdi,
            'DTR': dtr,
            'GSL': gsl,
            # Precipitation (11)
            'Rx1day': rx1day,
            'Rx5day': rx5day,
            'SDII': sdii,
            'PRCPTOT': prcptot,
            'R10mm': r10mm,
            'R20mm': r20mm,
            'Rnnmm': rnnmm,
            'R95pTOT': r95ptot,
            'R99pTOT': r99ptot,
            'CDD': cdd,
            'CWD': cwd
        })
        
    df_etccdi = pd.DataFrame(records)
    out_csv = TABLES_DIR / "etccdi_indices_annual_2000_2025.csv"
    df_etccdi.to_csv(out_csv, index=False)
    print(f"Calculated 27 ETCCDI indices saved to: {out_csv.name}")
    return df_etccdi

# -----------------------------------------------------------------------------
# 4. MANN-KENDALL TEST & SEN'S SLOPE ESTIMATION
# -----------------------------------------------------------------------------
def mann_kendall_test(x):
    """
    Menghitung statistik Mann-Kendall Z dan p-value untuk time-series x.
    """
    n = len(x)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])
            
    # Ties calculation
    unique_x, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    
    if var_s == 0:
        return 0.0, 1.0
        
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
        
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    return z, p_value

def compute_summary_and_trends(df_etccdi):
    print("\nComputing Mann-Kendall Trends & Sen's Slope for all indices...")
    years = df_etccdi['year'].values
    
    indices_info = [
        # Suhu
        ('TXx', 'Suhu Maksimum Tertinggi', 'C', 'Suhu'),
        ('TXn', 'Suhu Maksimum Terendah', 'C', 'Suhu'),
        ('TNx', 'Suhu Minimum Tertinggi', 'C', 'Suhu'),
        ('TNn', 'Suhu Minimum Terendah', 'C', 'Suhu'),
        ('SU', 'Summer Days (TX > 25 C)', 'hari', 'Suhu'),
        ('TR', 'Tropical Nights (TN > 20 C)', 'hari', 'Suhu'),
        ('FD', 'Frost Days (TN < 0 C)', 'hari', 'Suhu'),
        ('ID', 'Ice Days (TX < 0 C)', 'hari', 'Suhu'),
        ('TX10p', 'Cool Days (TX < 10th pct)', '%', 'Suhu'),
        ('TX90p', 'Warm Days (TX > 90th pct)', '%', 'Suhu'),
        ('TN10p', 'Cool Nights (TN < 10th pct)', '%', 'Suhu'),
        ('TN90p', 'Warm Nights (TN > 90th pct)', '%', 'Suhu'),
        ('WSDI', 'Warm Spell Duration Index', 'hari', 'Suhu'),
        ('CSDI', 'Cold Spell Duration Index', 'hari', 'Suhu'),
        ('DTR', 'Diurnal Temperature Range', 'C', 'Suhu'),
        ('GSL', 'Growing Season Length', 'hari', 'Suhu'),
        # Presipitasi
        ('Rx1day', 'Curah Hujan Maksimum 1 Hari', 'mm', 'Presipitasi'),
        ('Rx5day', 'Curah Hujan Maksimum 5 Hari', 'mm', 'Presipitasi'),
        ('SDII', 'Simple Daily Intensity Index', 'mm/hari', 'Presipitasi'),
        ('PRCPTOT', 'Total Presipitasi Hari Basah', 'mm', 'Presipitasi'),
        ('R10mm', 'Hari Hujan Lebat (>= 10 mm)', 'hari', 'Presipitasi'),
        ('R20mm', 'Hari Hujan Sangat Lebat (>= 20 mm)', 'hari', 'Presipitasi'),
        ('Rnnmm', 'Hari Hujan Ekstrem (nn=50 mm)', 'hari', 'Presipitasi'),
        ('R95pTOT', 'Total Hujan Sangat Basah (R95p)', 'mm', 'Presipitasi'),
        ('R99pTOT', 'Total Hujan Amat Sangat Basah (R99p)', 'mm', 'Presipitasi'),
        ('CDD', 'Hari Kering Berturut-turut', 'hari', 'Presipitasi'),
        ('CWD', 'Hari Basah Berturut-turut', 'hari', 'Presipitasi')
    ]
    
    summary_rows = []
    for code, name, unit, category in indices_info:
        y_vals = df_etccdi[code].values
        
        mean_val = np.mean(y_vals)
        std_val = np.std(y_vals, ddof=1)
        min_val = np.min(y_vals)
        max_val = np.max(y_vals)
        
        # Check if constant (like FD, ID, GSL)
        if np.all(y_vals == y_vals[0]):
            z_mk, p_val = 0.0, 1.0
            slope, intercept, low_slope, high_slope = 0.0, y_vals[0], 0.0, 0.0
        else:
            z_mk, p_val = mann_kendall_test(y_vals)
            res_slope = stats.theilslopes(y_vals, years)
            slope, intercept, low_slope, high_slope = res_slope
            
        slope_per_decade = slope * 10.0
        sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ("+" if p_val < 0.1 else "ns")))
        
        summary_rows.append({
            'Kategori': category,
            'Kode': code,
            'Nama_Indeks': name,
            'Satuan': unit,
            'Mean': round(mean_val, 2),
            'Std': round(std_val, 2),
            'Min': round(min_val, 2),
            'Max': round(max_val, 2),
            'MK_Z': round(z_mk, 3),
            'p_value': round(p_val, 4),
            'Signifikansi': sig_str,
            'Sens_Slope_Tahun': round(slope, 4),
            'Sens_Slope_Dekade': round(slope_per_decade, 3)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    out_csv = TABLES_DIR / "etccdi_summary_statistics_trends.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"Summary statistics and trend analysis saved to: {out_csv.name}")
    
    # Display table in terminal
    print("\n" + "="*90)
    print("RINGKASAN STATISTIK & TREN MANN-KENDALL 27 INDEKS ETCCDI KEBUMEN (2000-2025):")
    print("="*90)
    print(f"{'Kode':<8} | {'Kategori':<11} | {'Mean':>7} {'Satuan':<7} | {'Min':>7} {'Max':>7} | {'MK Z':>6} {'p-val':>7} {'Sig':>4} | {'Slope/Dekade':>12}")
    print("-" * 90)
    for _, r in df_summary.iterrows():
        print(f"{r['Kode']:<8} | {r['Kategori']:<11} | {r['Mean']:>7.2f} {r['Satuan']:<7} | {r['Min']:>7.2f} {r['Max']:>7.2f} | {r['MK_Z']:>6.2f} {r['p_value']:>7.4f} {r['Signifikansi']:>4} | {r['Sens_Slope_Dekade']:>10.3f} {r['Satuan']}/dekade")
    return df_summary

# -----------------------------------------------------------------------------
# 5. MULTI-SATELLITE PRECIPITATION INDICES INTER-COMPARISON
# -----------------------------------------------------------------------------
def compute_satellite_precipitation_comparison(df_sat, base_start=2001, base_end=2020):
    print("\nComputing Multi-Satellite Precipitation Inter-comparison...")
    products = ['ERA5_LAND', 'CHIRPS_RNL', 'GSMaP', 'IMERG']
    
    years = sorted(df_sat['year'].unique())
    sat_results = []
    
    for prod in products:
        if prod not in df_sat.columns:
            continue
            
        # Baseline wet days for percentiles
        base_df = df_sat[(df_sat['year'] >= base_start) & (df_sat['year'] <= base_end)]
        wet_base = base_df[base_df[prod] >= 1.0][prod]
        p95 = np.percentile(wet_base, 95)
        p99 = np.percentile(wet_base, 99)
        
        # Compute 5-day rolling sum
        prod_5day = df_sat[prod].rolling(5, min_periods=1).sum()
        
        prod_annual = []
        for yr in years:
            mask_yr = df_sat['year'] == yr
            y_pr = df_sat.loc[mask_yr, prod].values
            y_pr5 = prod_5day[mask_yr].values
            
            wet_m = y_pr >= 1.0
            wet_cnt = wet_m.sum()
            
            rx1 = np.max(y_pr)
            rx5 = np.max(y_pr5)
            sdii = np.sum(y_pr[wet_m]) / max(1, wet_cnt) if wet_cnt > 0 else 0
            prcptot = np.sum(y_pr[wet_m])
            r10 = np.sum(y_pr >= 10.0)
            r20 = np.sum(y_pr >= 20.0)
            rnn = np.sum(y_pr >= 50.0)
            r95p = np.sum(y_pr[wet_m & (y_pr > p95)])
            r99p = np.sum(y_pr[wet_m & (y_pr > p99)])
            cdd = calculate_max_consecutive(y_pr < 1.0)
            cwd = calculate_max_consecutive(y_pr >= 1.0)
            
            prod_annual.append({
                'Product': prod,
                'Year': yr,
                'Rx1day': rx1,
                'Rx5day': rx5,
                'SDII': sdii,
                'PRCPTOT': prcptot,
                'R10mm': r10,
                'R20mm': r20,
                'Rnnmm': rnn,
                'R95pTOT': r95p,
                'R99pTOT': r99p,
                'CDD': cdd,
                'CWD': cwd
            })
            
        sat_results.extend(prod_annual)
        
    df_sat_res = pd.DataFrame(sat_results)
    out_csv = TABLES_DIR / "satellite_precipitation_indices_comparison.csv"
    df_sat_res.to_csv(out_csv, index=False)
    print(f"Multi-Satellite precipitation indices comparison saved to: {out_csv.name}")
    
    # Compute multi-year mean by product
    mean_comp = df_sat_res.groupby('Product').mean(numeric_only=True).drop(columns=['Year']).round(2)
    mean_csv = TABLES_DIR / "satellite_precipitation_climatology_mean.csv"
    mean_comp.to_csv(mean_csv)
    print(f"Product Climatological Mean Table:\n{mean_comp}")
    return df_sat_res

if __name__ == "__main__":
    daily_df = load_and_aggregate_daily_era5()
    df_etccdi = compute_all_27_etccdi(daily_df)
    df_summary = compute_summary_and_trends(df_etccdi)
    
    df_sat = load_satellite_precipitation()
    df_sat_res = compute_satellite_precipitation_comparison(df_sat)
    print("\n[SUCCESS] Phase 1 ETCCDI Computation completed successfully!")
