import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI GLOBAL
# ==========================================
# Pemetaan variabel dari IoT ke ERA5
era5_mapping = {
    'temperature': 'temperature_era5', 
    'humidity': 'humidity_era5',
    'pressure': 'sealevel_pressure_era5',
    'dew': 'dewpoint_era5',
    'rainrate': 'rain_mm',
}

# Rentang Waktu (Sesuai Permintaan dalam UTC)
START_DATE = "2025-01-01 00:00:00"
END_DATE   = "2026-05-31 23:59:59"

# Penyesuaian Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IOT_RAW_DIR = os.path.join(BASE_DIR, 'Analisis_Meteorologi', 'cache_data')

# Path satelit
ERA5_PATH = os.path.join(BASE_DIR, 'Google_Earth_Engine', 'Data_Satelit', 'ERA5_Hourly_All_Requested_Features_2000_2026.csv')
IMERG_PATH = os.path.join(BASE_DIR, 'Google_Earth_Engine', 'Data_Satelit', 'Rainfall_IMERG_TimeSeries_UNIX.csv')
GSMAP_PATH = os.path.join(BASE_DIR, 'Google_Earth_Engine', 'Data_Satelit', 'Rainfall_GSMaP_TimeSeries_UNIX.csv')
OYA_PATH = os.path.join(BASE_DIR, 'Google_Earth_Engine', 'Data_Satelit', 'Rainfall_Oya_TimeSeries_UNIX.csv')

OUTPUT_DIR = os.path.join(BASE_DIR, 'Google_Earth_Engine', 'Data_Satelit')

# Parameter Hujan
BATAS_MAKSIMAL_PER_MENIT = 3.0
KONSTANTA_TIP = 0.3


# ==========================================
# 2. FUNGSI PEMUATAN DATA
# ==========================================

def load_iot_data(station_id="id-05"):
    """Memuat dan melakukan agregasi awal data IoT AWS menjadi format hourly."""
    print(f"🔄 Membaca data raw IoT untuk stasiun {station_id}...")
    file_path = os.path.join(IOT_RAW_DIR, f"{station_id}_raw.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data raw untuk {station_id} tidak ditemukan di {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Konversi ke UTC
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.sort_values('timestamp').set_index('timestamp')
        
    # --- Pra-pemrosesan Hujan (Increment & Outlier) ---
    if 'rainrate' in df.columns:
        df['delta_raw'] = df['rainrate'].diff()
        
        # Jika delta negatif (sensor reset), ambil nilai bacaan asli, jika tidak ambil selisihnya
        df['actual_increment'] = np.where(
            df['delta_raw'] < 0, 
            df['rainrate'],      
            df['delta_raw']      
        )
        df['actual_increment'] = df['actual_increment'].fillna(0).clip(lower=0)

        # Hancurkan lonjakan tak wajar (outlier)
        kondisi_outlier = df['actual_increment'] > BATAS_MAKSIMAL_PER_MENIT
        df.loc[kondisi_outlier, 'actual_increment'] = 0.0

        # Standarisasi pembulatan tipping bucket
        df['tips_count'] = np.round(df['actual_increment'] / KONSTANTA_TIP)
        df['increment_fixed'] = df['tips_count'] * KONSTANTA_TIP

        # Rekonstruksi Akumulasi Per Jam
        df['rainrate'] = df.groupby(df.index.floor('h'))['increment_fixed'].cumsum()
        
    # Agregasi ke Resolusi 1 Jam (Hourly)
    agg_dict = {}
    if 'rainrate' in df.columns: agg_dict['rainrate'] = 'max'
    if 'temperature' in df.columns: agg_dict['temperature'] = 'mean'
    if 'humidity' in df.columns: agg_dict['humidity'] = 'mean'
    if 'pressure' in df.columns: agg_dict['pressure'] = 'mean'
    if 'dew' in df.columns: agg_dict['dew'] = 'mean'
    if 'volt' in df.columns: agg_dict['volt'] = 'mean'
    if 'lux' in df.columns: agg_dict['lux'] = 'mean'
    
    df_hourly = df.resample('h').agg(agg_dict)
    
    # Pastikan zona waktu sesuai sebelum lanjut
    if df_hourly.index.tz is None:
        df_hourly.index = df_hourly.index.tz_localize('UTC')
    else:
        df_hourly.index = df_hourly.index.tz_convert('UTC')
        
    print(f"✅ Data IoT {station_id} siap (Dimensi: {df_hourly.shape})")
    return df_hourly


def load_era5_data():
    """Memuat data satelit ERA5 reference."""
    print("🔄 Membaca data satelit ERA5...")
    df_era5 = pd.read_csv(ERA5_PATH)
    
    # Konversi waktu ke UTC agar sinkron
    df_era5['timestamp'] = pd.to_datetime(df_era5['datetime_utc'], utc=True)
    df_era5 = df_era5.set_index('timestamp')
    
    print(f"✅ Data ERA5 siap (Dimensi: {df_era5.shape})")
    return df_era5


def load_multi_satellite_rain(start_time, end_time):
    """Memuat data curah hujan dari IMERG, GSMaP, dan ERA5 untuk dihitung mediannya."""
    print("🔄 Memproses konsensus median hujan dari ERA5, IMERG, dan GSMaP...")
    
    master_index = pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')
    
    # 1. IMERG
    df_imerg = pd.read_csv(IMERG_PATH)
    df_imerg['timestamp'] = pd.to_datetime(df_imerg['datetime_utc'], utc=True)
    df_imerg_hourly = df_imerg.set_index('timestamp').resample('1h').sum().reindex(master_index)['precipitation_mmhr']
    
    # 2. GSMaP
    df_gsmap = pd.read_csv(GSMAP_PATH)
    df_gsmap['timestamp'] = pd.to_datetime(df_gsmap['datetime_utc'], utc=True)
    df_gsmap_hourly = df_gsmap.set_index('timestamp').resample('1h').sum().reindex(master_index)['hourlyPrecipRateGC']
    
    # 3. ERA5
    df_era5 = pd.read_csv(ERA5_PATH)
    df_era5['timestamp'] = pd.to_datetime(df_era5['datetime_utc'], utc=True)
    df_era5_hourly = df_era5.set_index('timestamp').reindex(master_index)['rain_mm']
    
    # Gabungkan & Hitung Median
    df_rain_sat = pd.DataFrame({
        'imerg': df_imerg_hourly,
        'gsmap': df_gsmap_hourly,
        'era5': df_era5_hourly
    })
    
    # Nilai Tengah (Median)
    median_rain = df_rain_sat.median(axis=1).fillna(0.0)
    print(f"✅ Konsensus Hujan Satelit Siap (Nilai Maksimal Median: {median_rain.max():.2f} mm)")
    return median_rain


# ==========================================
# 3. PIPELINE IMPUTASI HIBRIDA
# ==========================================

def bersihkan_data_hourly(df_hourly, df_era5_reference, df_rain_median, start_time, end_time):
    """
    Melakukan imputasi hibrida persis sesuai logika notebook:
    1. Hampel Filter (Outlier)
    2. PCHIP Interpolation (Lubang <= 4 Jam)
    3. Substitusi ERA5 dengan Bias Correction (MBE & MAE)
    4. Substitusi Median Satelit (Khusus Hujan/Rainrate)
    """
    print(f"\n🚀 Memulai Pipeline Imputasi Hibrida | Rentang: {start_time} s.d {end_time}")
    
    # 1. Reindexing Mutlak (Memastikan tidak ada jam yang terlewat)
    master_index = pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')
    df_reindexed = df_hourly.reindex(master_index)
    df_reindexed.index.name = 'timestamp'
    
    df_raw = df_reindexed.copy()
    
    kolom_sensor = ['temperature', 'humidity', 'pressure', 'dew', 'lux', 'volt']
    window = 12
    n_sigmas = 3
    
    for col in kolom_sensor:
        if col not in df_reindexed.columns: continue
        
        # Buat kolom flag imputasi
        qc_col = f'is_imputed_{col}'
        df_reindexed[qc_col] = df_reindexed[col].isnull().astype(int)
        
        # A. Hampel Filter (Deteksi Outlier)
        rolling_median = df_reindexed[col].rolling(window=window, center=True).median()
        deviasi = np.abs(df_reindexed[col] - rolling_median)
        mad = deviasi.rolling(window=window, center=True).median()
        threshold = n_sigmas * 1.4826 * mad
        outlier_idx = deviasi > threshold
        
        # Hancurkan outlier
        df_reindexed.loc[outlier_idx, col] = np.nan
        df_reindexed.loc[outlier_idx, qc_col] = 1 
        
        # B. Imputasi Matematika (PCHIP) - Maksimal 4 jam
        df_reindexed[col] = df_reindexed[col].interpolate(method='pchip', limit=4, limit_direction='forward')
        
        # C. Asimilasi ERA5 untuk Sisa Lubang
        nama_kolom_era5 = era5_mapping.get(col)
        if nama_kolom_era5 and nama_kolom_era5 in df_era5_reference.columns:
            # Hitung bias agar kurva satelit menyesuaikan elevasi stasiun IoT
            valid_mask = df_reindexed[col].notnull() & df_era5_reference[nama_kolom_era5].notnull()
            if valid_mask.sum() > 0:
                iot_valid = df_reindexed.loc[valid_mask, col]
                era5_valid = df_era5_reference.loc[valid_mask, nama_kolom_era5]
                
                bias = (iot_valid - era5_valid).mean()
                mae = (iot_valid - era5_valid).abs().mean()
                print(f"   -> [Bias Correction] {col.upper():<12}: MBE = {bias:>+7.2f} | MAE = {mae:>5.2f}")
                era5_terkoreksi = df_era5_reference[nama_kolom_era5] + bias
            else:
                era5_terkoreksi = df_era5_reference[nama_kolom_era5]
                
            # Tambal sisanya
            df_reindexed[col] = df_reindexed[col].fillna(era5_terkoreksi)
            
        # Tambalan darurat terakhir jika ERA5 tidak tersedia
        df_reindexed[col] = df_reindexed[col].bfill().ffill()
        
        # D. Batasan Fisika
        if col == 'humidity': 
            df_reindexed[col] = df_reindexed[col].clip(0, 100)
        elif col == 'lux':
            df_reindexed[col] = df_reindexed[col].clip(lower=0)
            
    # --- Penanganan Khusus Hujan (Asimilasi Median Satelit) ---
    col = 'rainrate'
    if col in df_reindexed.columns:
        qc_col = f'is_imputed_{col}'
        df_reindexed[qc_col] = df_reindexed[col].isnull().astype(int)
        
        # Tambal hujan IoT yang bolong menggunakan data median satelit
        print(f"   -> [Rainfall Injection] Menginjeksi data hujan dari Konsensus Satelit (Median)...")
        df_reindexed[col] = df_reindexed[col].fillna(df_rain_median)
            
        # Jika masih ada yang kosong, anggap tidak hujan (0.0)
        df_reindexed[col] = df_reindexed[col].fillna(0.0)

    print("✅ Pipeline Imputasi Selesai!")
    return df_raw, df_reindexed


# ==========================================
# 4. VALIDASI DAN EKSPOR
# ==========================================

def save_hourly_clean_dataset(df_clean, station_id="id-05", output_folder="clear data"):
    """Memvalidasi integritas data dan mengekspor hasilnya ke CSV dengan format Markdown audit."""
    os.makedirs(output_folder, exist_ok=True)
    
    df_final = df_clean.copy()
    
    # Standarisasi kolom Dewpoint
    if 'dew' in df_final.columns:
        df_final = df_final.rename(columns={'dew': 'dewpoint'})
        
    if not isinstance(df_final.index, pd.DatetimeIndex):
        df_final.index = pd.to_datetime(df_final.index)
        
    df_final['unixtime'] = df_final.index.astype('int64') // 10**9
    df_final['datetime_utc'] = df_final.index.strftime('%Y-%m-%d %H:%M:%S')
    
    kolom_wajib = ['datetime_utc', 'unixtime', 'temperature', 'humidity', 'pressure', 'dewpoint', 'rainrate']
    for col in kolom_wajib:
        if col not in df_final.columns:
            df_final[col] = np.nan
            
    df_final = df_final[kolom_wajib]
    
    # Datetime Validation
    jml_duplikat = df_final.index.duplicated().sum()
    is_monotonic = df_final.index.is_monotonic_increasing
    start_dt = df_final.index.min()
    end_dt = df_final.index.max()
    
    expected_hours = int((end_dt - start_dt).total_seconds() // 3600) + 1 if pd.notnull(start_dt) else 0
    actual_hours = len(df_final)
    missing_hours = expected_hours - actual_hours if expected_hours > 0 else 0
    
    invalid_dt = df_final['datetime_utc'].isnull().sum()
    invalid_unix = df_final['unixtime'].isnull().sum()
    
    # Numeric Validation
    numeric_cols = ['temperature', 'humidity', 'pressure', 'dewpoint', 'rainrate']
    nan_counts = df_final[numeric_cols].isna().sum().to_dict()
    inf_counts = np.isinf(df_final[numeric_cols]).sum().to_dict()
    
    temp_valid = df_final['temperature'].between(-10, 60) | df_final['temperature'].isna()
    hum_valid = df_final['humidity'].between(0, 100) | df_final['humidity'].isna()
    pres_valid = df_final['pressure'].between(850, 1100) | df_final['pressure'].isna()
    dew_valid = df_final['dewpoint'].between(-20, 60) | df_final['dewpoint'].isna()
    
    temp_outbounds = (~temp_valid).sum()
    hum_outbounds = (~hum_valid).sum()
    pres_outbounds = (~pres_valid).sum()
    dew_outbounds = (~dew_valid).sum()
    
    output_path = os.path.join(output_folder, f"{station_id}_clear_data_hourly.csv")
    df_final.reset_index(drop=True).to_csv(output_path, index=False)
    
    report_md = f"""
### 📊 Summary Audit Preprocessing & Validasi Hibrida: `{station_id}`

#### Output Files Generated
- **CSV Export:** `{output_path}`
- **Total Records:** `{actual_hours:,}`
- **Time Range:** `{start_dt}` to `{end_dt}`

#### Datetime Validation
- **Missing Timestamps (Hours):** `{missing_hours}`
- **Duplicate Timestamps:** `{jml_duplikat}`
- **Chronological Ordering Verified:** `{'Yes' if is_monotonic else 'No'}`
- **Invalid Datetime Values:** `{invalid_dt}`
- **Invalid Unix Timestamps:** `{invalid_unix}`

#### Numeric & Meteorological Validation
| Metric | NaN Count | Infinity Count | Out of Bounds Count |
|--------|-----------|----------------|---------------------|
| Temperature | {nan_counts['temperature']} | {inf_counts['temperature']} | {temp_outbounds} |
| Humidity | {nan_counts['humidity']} | {inf_counts['humidity']} | {hum_outbounds} |
| Pressure | {nan_counts['pressure']} | {inf_counts['pressure']} | {pres_outbounds} |
| Dewpoint | {nan_counts['dewpoint']} | {inf_counts['dewpoint']} | {dew_outbounds} |
| Rainrate | {nan_counts['rainrate']} | {inf_counts['rainrate']} | 0 |
"""
    print(report_md)
    return output_path


# ==========================================
# 5. VISUALISASI HASIL IMPUTASI DAN FORENSIK
# ==========================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import missingno as msno

def setup_plot_dir(station_id):
    plot_dir = os.path.join(OUTPUT_DIR, "plots_imputasi", station_id)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

# --- 5A. KOMPARASI (SIDIK JARI ASIMILASI) ---
def plot_komparasi(df_raw, df_clean, df_era5, df_rain_median, start_time, end_time, station_id="id-05"):
    print(f"\n📊 Membuat plot visualisasi komparasi (Sidik Jari Asimilasi): {start_time} s.d {end_time}")
    plot_dir = setup_plot_dir(station_id)
    kolom_visual = ['temperature', 'humidity', 'pressure', 'dew', 'rainrate']

    df_raw_plot = df_raw.loc[start_time:end_time]
    df_clean_plot = df_clean.loc[start_time:end_time]
    df_era5_plot = df_era5.loc[start_time:end_time]
    df_rain_median_plot = df_rain_median.loc[start_time:end_time]

    for col in kolom_visual:
        plt.figure(figsize=(15, 5))
        
        # Plot data satelit / referensi
        if col == 'rainrate':
            plt.plot(df_rain_median_plot.index, df_rain_median_plot, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Satelit Rain Consensus (Median)')
        else:
            col_era5 = era5_mapping.get(col)
            if col_era5 and col_era5 in df_era5_plot.columns:
                plt.plot(df_era5_plot.index, df_era5_plot[col_era5], color='green', linestyle=':', linewidth=2, alpha=0.7, label='Satelit ERA5')

        # Plot data raw
        if col in df_raw_plot.columns:
            plt.plot(df_raw_plot.index, df_raw_plot[col], color='red', marker='x', markersize=5, linestyle='None', alpha=0.6, label='Raw Data Sensor')

        # Plot data clean
        if col in df_clean_plot.columns:
            plt.plot(df_clean_plot.index, df_clean_plot[col], color='orange', linewidth=1.5, alpha=0.9, label='Clean Data (Imputasi Hibrida)')

        plt.title(f'Sidik Jari Asimilasi Data {station_id}: {col.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel(f'Nilai {col.capitalize()}')
        plt.xlabel('Waktu')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"1_komparasi_{col}.png"), dpi=150)
        plt.close()

# --- 5B. SEBELUM VS SESUDAH ---
def plot_sebelum_sesudah(df_raw, df_clean, start_time, end_time, station_id="id-05"):
    print(f"📊 Membuat plot Sebelum vs Sesudah: {start_time} s.d {end_time}")
    plot_dir = setup_plot_dir(station_id)
    kolom_visual = ['temperature', 'humidity', 'pressure', 'dew', 'rainrate']
    
    df_raw_plot = df_raw.loc[start_time:end_time]
    df_clean_plot = df_clean.loc[start_time:end_time]

    for col in kolom_visual:
        plt.figure(figsize=(15, 6))
        if col in df_clean_plot.columns:
            plt.plot(df_clean_plot.index, df_clean_plot[col], color='blue', linewidth=1.5, label='Sesudah (Tertambal Satelit)', zorder=1)
        if col in df_raw_plot.columns:
            plt.scatter(df_raw_plot.index, df_raw_plot[col], color='red', label='Sebelum (Data IoT Asli)', marker='o', s=15, zorder=2)

        plt.title(f'Efek Pembersihan & Asimilasi Data: {col.capitalize()} ({start_time} s.d {end_time})', fontsize=14, fontweight='bold')
        plt.ylabel(f'{col.capitalize()}')
        plt.xlabel('Waktu')
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"2_before_after_{col}.png"), dpi=150)
        plt.close()

# --- 5C. TREN & HIETOGRAF ---
def plot_station_trends(df_clean, freq='d', agg_method='mean', station_id="id-05"):
    print(f"📊 Membuat plot Trend & Hietograf ({freq} | {agg_method})...")
    plot_dir = setup_plot_dir(station_id)
    kolom_visual = ['temperature', 'humidity', 'pressure', 'dewpoint']
    
    temp_df = df_clean.copy()
    if temp_df.index.name == 'timestamp': temp_df = temp_df.reset_index()
    time_col = f'time_group_{freq}'
    
    if freq == 'd':
        temp_df[time_col] = temp_df['timestamp'].dt.floor('d')
        xlabel = 'Tanggal'
        label_freq = 'Harian'
    else:
        temp_df[time_col] = temp_df['timestamp'].dt.floor('h')
        xlabel = 'Waktu (Jam)'
        label_freq = 'Per Jam'
        
    for col in kolom_visual:
        if col not in temp_df.columns: continue
        trend_data = temp_df.groupby(time_col)[col].agg(agg_method).dropna().reset_index()
        
        plt.figure(figsize=(15, 7))
        plt.plot(trend_data[time_col], trend_data[col], label=station_id, alpha=0.8, color='purple')
        plt.xlabel(xlabel)
        plt.ylabel(f"Rata-rata {col.capitalize()}")
        plt.title(f"Perbandingan Rata-rata {col.capitalize()} ({label_freq})", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"3_trend_{freq}_{col}.png"), dpi=150)
        plt.close()

    # HIETOGRAF KHUSUS HUJAN
    if 'rainrate' in temp_df.columns:
        trend_data = temp_df.groupby(time_col)['rainrate'].agg('max').dropna().reset_index()
        plt.figure(figsize=(15, 6))
        plt.plot(trend_data[time_col], trend_data['rainrate'], color='dodgerblue', lw=1)
        plt.fill_between(trend_data[time_col], trend_data['rainrate'], color='dodgerblue', alpha=0.3)
        plt.title(f'Hietograf: Distribusi Curah Hujan ({label_freq})', fontsize=16, fontweight='bold')
        plt.ylabel('Curah Hujan (mm)')
        plt.xlabel(xlabel)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        plt.xticks(rotation=45)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"3_hietograf_{freq}_rainrate.png"), dpi=150)
        plt.close()

# --- 5D. MATRIKS KORELASI (PEARSON & SPEARMAN) & SCATTER ---
def cek_korelasi_transparansi(df_raw, df_clean, df_era5, df_rain_median, start_time, end_time, station_id="id-05"):
    print(f"📊 Menghitung Matriks Korelasi (Pearson & Spearman) & Scatter Plot...")
    plot_dir = setup_plot_dir(station_id)
    kolom_visual = ['temperature', 'humidity', 'pressure', 'dew', 'rainrate']
    
    df_raw_plot = df_raw.loc[start_time:end_time]
    df_clean_plot = df_clean.loc[start_time:end_time]
    df_era5_plot = df_era5.loc[start_time:end_time]
    df_rain_median_plot = df_rain_median.loc[start_time:end_time]

    for col in kolom_visual:
        df_korelasi = pd.DataFrame(index=df_clean_plot.index)
        
        if col == 'rainrate':
            df_korelasi['Satelit'] = df_rain_median_plot.reindex(df_clean_plot.index)
        else:
            col_era5 = era5_mapping.get(col)
            if col_era5 and col_era5 in df_era5_plot.columns:
                df_korelasi['Satelit'] = df_era5_plot[col_era5].reindex(df_clean_plot.index)
            
        if col in df_raw_plot.columns: 
            df_korelasi['Raw_IoT'] = df_raw_plot[col]
            
        if col in df_clean_plot.columns: 
            df_korelasi['Clean_IoT'] = df_clean_plot[col]
        
        # Tampilkan persentase NaN di IoT Mentah (Ini untuk membuktikan diagnosis kita!)
        if 'Raw_IoT' in df_korelasi.columns:
            persen_nan = df_korelasi['Raw_IoT'].isnull().mean() * 100
            print(f"WARNING: Fakta Lapangan: Data IoT Mentah Anda kosong sebanyak {persen_nan:.2f}% pada rentang ini.")
        
        # Pearson & Spearman Heatmap
        for method in ['pearson', 'spearman']:
            corr_matrix = df_korelasi.corr(method=method)
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, fmt=".3f", linewidths=1, linecolor='black')
            plt.title(f'Matriks Korelasi ({method.capitalize()}): {col.capitalize()}', fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"4_korelasi_heatmap_{method}_{col}.png"), dpi=150)
            plt.close()
        
        # Scatter Plot 1:1
        if 'Satelit' in df_korelasi.columns and 'Clean_IoT' in df_korelasi.columns:
            plt.figure(figsize=(7, 7))
            plt.scatter(df_korelasi['Satelit'], df_korelasi['Clean_IoT'], alpha=0.5, color='blue')
            max_val = max(df_korelasi['Satelit'].max(), df_korelasi['Clean_IoT'].max())
            min_val = min(df_korelasi['Satelit'].min(), df_korelasi['Clean_IoT'].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Garis Identik (1:1)')
            plt.title(f'Scatter Plot: Satelit vs IoT Bersih ({col.capitalize()})')
            plt.xlabel('Nilai Satelit')
            plt.ylabel('Nilai IoT Bersih')
            plt.legend()
            plt.grid(True, linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"4_korelasi_scatter_{col}.png"), dpi=150)
            plt.close()

# --- 5E. ANALISIS FORENSIK (MISSINGNO DLL) ---
def forensic_analysis(df_raw, station_id="id-05"):
    print(f"📊 Menghasilkan Visualisasi Forensik (Missingno)...")
    plot_dir = setup_plot_dir(station_id)
    
    # Missingno Matrix
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    msno.matrix(df_raw, ax=ax, sparkline=False, fontsize=10)
    ax.set_title(f"Missingno Matrix Plot - {station_id}", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"5_forensic_matrix.png"), dpi=150)
    plt.close()
    
    # Missingno Heatmap
    if df_raw.isnull().sum().sum() > 0:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        msno.heatmap(df_raw, ax=ax, fontsize=12)
        ax.set_title(f"Missingno Heatmap (Correlation of Missingness) - {station_id}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"5_forensic_heatmap.png"), dpi=150)
        plt.close()
        
    # Daily Completeness
    if 'temperature' in df_raw.columns:
        daily_completeness = df_raw['temperature'].notnull().resample('d').mean() * 100
        plt.figure(figsize=(15, 4))
        plt.plot(daily_completeness.index, daily_completeness.values, color='green', lw=2)
        plt.fill_between(daily_completeness.index, daily_completeness.values, 100, color='red', alpha=0.3, label='Missing Data Area')
        plt.title(f"Daily Completeness Percentage - {station_id}", fontsize=14, fontweight='bold')
        plt.ylabel("Completeness (%)")
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"5_forensic_completeness.png"), dpi=150)
        plt.close()
        
        # Missing Timeline Gantt Chart
        plt.figure(figsize=(15, 2))
        missing_series = df_raw['temperature'].isnull().astype(int)
        plt.fill_between(missing_series.index, 0, missing_series, where=missing_series==1, color='red', alpha=0.8)
        plt.title(f"Missing Timeline (Red = Outage) - {station_id}", fontsize=14, fontweight='bold')
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"5_forensic_gantt.png"), dpi=150)
        plt.close()


def main():
    print("="*60)
    print("  SKRIP PEMBERSIHAN DATA & IMPUTASI HIBRIDA (ERA5 + MULTI-SATELIT)")
    print("="*60)
    
    try:
        # 1. Pemuatan Data
        df_iot = load_iot_data("id-05")
        df_era5 = load_era5_data()
        df_rain_median = load_multi_satellite_rain(START_DATE, END_DATE)
        
        # 2. Eksekusi Pipeline
        df_raw_reindexed, df_bersih = bersihkan_data_hourly(
            df_hourly=df_iot,
            df_era5_reference=df_era5,
            df_rain_median=df_rain_median,
            start_time=START_DATE,
            end_time=END_DATE
        )
        
        # 3. Validasi & Ekspor
        save_hourly_clean_dataset(df_bersih, "id-05", OUTPUT_DIR)
        
        # 4. Visualisasi Hasil & Forensik Lengkap
        start_plot = "2025-01-01 00:00:00"
        end_plot = "2025-01-14 23:59:59"
        
        forensic_analysis(df_iot, "id-05")
        plot_komparasi(df_raw_reindexed, df_bersih, df_era5, df_rain_median, start_plot, end_plot, "id-05")
        plot_sebelum_sesudah(df_raw_reindexed, df_bersih, start_plot, end_plot, "id-05")
        plot_station_trends(df_bersih, freq='h', agg_method='mean', station_id="id-05")
        plot_station_trends(df_bersih, freq='d', agg_method='mean', station_id="id-05")
        cek_korelasi_transparansi(df_raw_reindexed, df_bersih, df_era5, df_rain_median, START_DATE, END_DATE, "id-05")
        
        print(f"\n✅ SELURUH PROSES SELESAI. Cek folder 'Google_Earth_Engine/Data_Satelit/plots_imputasi/id-05' untuk melihat visualisasi.")
        
    except Exception as e:
        import traceback
        print(f"\n❌ TERJADI KESALAHAN: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
