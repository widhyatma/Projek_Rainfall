# Reusable Rules & Methodology: Universal Meteorological Imputation Framework

Dokumen ini mendokumentasikan seluruh arsitektur, batasan fisika (*physics-informed constraints*), penanganan database SQLite, dan sinkronisasi Firebase yang telah dipelajari dan diintegrasikan ke dalam repository.

---

## 1. Arsitektur Imputasi Universal 2-Fase (`Model_Imputasi`)

1. **Strategi 2-Fase Berdasarkan Durasi Gap**:
   - **Fase 1 (Micro-Gaps $\le 15$ menit)**: Interpolasi Polinomial / Spline derajat 2 untuk efisiensi komputasi instan dan kontinuitas lokal tinggi.
   - **Fase 2 (Macro-Gaps $> 15$ menit hingga multiday outage)**: Multi-Input Denoising Bi-LSTM Autoencoder yang dikondisikan pada baseline makro ERA5 hasil interpolasi dan fitur waktu siklik ($\sin/\cos$ jam, hari).
2. **Topologi Jaringan Multi-Input**:
   - **Stream Kontinu**: `(batch, 120, 11)` yang memuat 4 variabel target ($T, RH, P, T_{\text{dew}}$) + 4 encoding siklik waktu + 3 variabel makro ERA5.
   - **Stream Kategorikal**: `Node_ID` stasiun di-embed via `layers.Embedding(num_nodes=8, embedding_dim=4)` dan diperluas via `RepeatVector(120)` agar satu model universal mampu menangani iklim mikro heterogen di berbagai node.
   - **Backbone RNN**: `Bidirectional(LSTM(64))` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Bidirectional(LSTM(32))` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `TimeDistributed(Dense(4))`.
3. **Loss Function Mask-Weighted MSE**:
   - Error hanya dihitung pada posisi data valid ground-truth (`tf.math.is_finite(y_true)`), sehingga jaringan mampu belajar merekonstruksi pola atmosfer meskipun data latihan di Firebase/SQLite memiliki *missingness*.

---

## 2. Batasan Fisika Termodinamika & Penghalusan (*De-Jittering*)

1. **Konsistensi Magnus-Tetens**:
   - Titik embun tidak boleh melampaui suhu ambien ($T_{\text{dew}} \le T_{\text{ambient}}$).
   - Penegakan: `df['dewpoint'] = np.minimum(df['dewpoint'], df['temperature'])`.
2. **Batasan Fisika Kelembaban Relatif**:
   - Kelembaban udara dibatasi pada rentang fisik atmosfer permukaan: $10\% \le RH \le 100\%$.
3. **Filter Savitzky-Golay Zero-Phase**:
   - Parameter: $W=15$ menit, $\text{order}=2$.
   - Menghilangkan *quantization stepping / staircasing noise* dari sensor digital (DHT22 / ADC) tanpa menggeser fase waktu maupun meredam puncak diurnal atau penurunan mendadak akibat hujan.

---

## 3. Protokol Query SQLite Database (`meteo_local_cache.db`)

1. **Tabel & Kolom**:
   - Tabel: `aws_measurements`.
   - Kolom: `station_id`, `unix_ts`, `temperature`, `humidity`, `pressure`, `dew_point`, `dew`, `volt`.
2. **Filter Stasiun Stabil**:
   - Melatih model menggunakan node stasiun luar ruangan yang stabil dan terkalibrasi (`id-03` dan `id-05`). Node tidak stabil seperti `id-01` di-exclude dari pelatihan.
3. **Deduplikasi Kolom & Unifikasi Dewpoint**:
   - Unifikasi kolom `dew_point` dan `dew` menjadi satu kolom `dewpoint` tunggal sebelum diindeks ke DataFrame untuk menghindari *duplicate column name* yang menyebabkan error *2D boolean indexing* pada saat Quality Control.
4. **Konversi Waktu & Grid 1-Menit**:
   - Konversi UTC unix timestamp ke waktu lokal: `pd.to_datetime(df['unix_ts'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)`.
   - Resampling ke grid kontinu 1-menitan (`df.resample('1min').mean()`).
5. **Standar UNIX Timestamp 10-Digit**:
   - Output timestamp harus selalu bertipe integer 10-digit detik:
     `((pd.to_datetime(df.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') - 7 * 3600).astype(int)`.

---

## 4. Protokol Sinkronisasi Firebase Realtime Database

1. **Pembersihan Record Sub-Menit (*Duplicate Purge*)**:
   - Sebelum mengunggah hasil rekonstruksi 1-menitan, data pada detik selain `:00` (seperti `:04`, `:18`, `:35` dari uji coba dalam ruangan) harus dihapus dari Firebase. Hal ini mencegah grafik visualisasi di dashboard berosilasi membentuk gelombang kotak (*sawtooth/square-wave*).
2. **Penulisan Multi-Path Atomic**:
   - Mengunggah data bersih terstruktur per-menit menggunakan atomic update `db.reference('/auto_weather_stat/{node_id}/data').update(payload)`.
