"""
High-Performance CSV to Excel (.xlsx) Converter
================================================
Mengonversi seluruh file CSV di folder Google_Earth_Engine/Data_Satelit
menjadi format Excel (.xlsx) menggunakan engine xlsxwriter constant_memory=True.
"""

import os
import glob
import time
import tempfile
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data_Satelit")

# Redirect Python tempdir to D: drive
CUSTOM_TMP_DIR = os.path.join(BASE_DIR, "tmp_temp")
os.makedirs(CUSTOM_TMP_DIR, exist_ok=True)
tempfile.tempdir = CUSTOM_TMP_DIR
os.environ['TMP'] = CUSTOM_TMP_DIR
os.environ['TEMP'] = CUSTOM_TMP_DIR


def convert_all_csv_to_excel(data_dir=DATA_DIR):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print("================================================================")
    print(f"[START] MEMULAI KONVERSI HIGH-PERFORMANCE CSV KE EXCEL (.xlsx)")
    print(f"[INFO] Path TempDir Ditargetkan ke D: Drive: {CUSTOM_TMP_DIR}")
    print(f"[INFO] Total File CSV Ditemukan: {len(csv_files)}")
    print("================================================================")
    
    for idx, csv_path in enumerate(csv_files, 1):
        filename = os.path.basename(csv_path)
        xlsx_name = os.path.splitext(filename)[0] + ".xlsx"
        xlsx_path = os.path.join(data_dir, xlsx_name)
        
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"\n[{idx}/{len(csv_files)}] Memproses: {filename} ({file_size_mb:.2f} MB)...")
        
        t0 = time.time()
        try:
            df = pd.read_csv(csv_path)
            rows = len(df)
            cols = len(df.columns)
            print(f"   - Dibaca: {rows:,} baris, {cols} kolom.")
            
            # Use xlsxwriter engine with constant_memory=True for fast streaming & low RAM
            with pd.ExcelWriter(xlsx_path, engine='xlsxwriter', engine_kwargs={'options': {'constant_memory': True}}) as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
                
            out_size_mb = os.path.getsize(xlsx_path) / (1024 * 1024)
            t_elapsed = time.time() - t0
            print(f"   [SUCCESS] Tersimpan ke: {xlsx_name} ({out_size_mb:.2f} MB) dalam {t_elapsed:.1f}s!")
            del df
        except Exception as e:
            print(f"   [ERROR] Gagal mengonversi {filename}: {e}")
            
    print("\n================================================================")
    print("[SUCCESS] SELURUH KONVERSI CSV KE EXCEL (.xlsx) SELESAI!")
    print("================================================================")


if __name__ == "__main__":
    convert_all_csv_to_excel()
