# Dokumentasi Pipeline Inferensi & Arsitektur Model Prediksi Curah Hujan

Dokumen ini menjelaskan secara menyeluruh rancangan sistem, rekayasa fitur, arsitektur pemodelan dua tahap (*two-stage hurdle modeling*), strategi adaptasi domain (*transfer learning/fine-tuning*), serta kerangka evaluasi hidrometeorologi terpadu untuk prediksi curah hujan jam-jaman (*hourly precipitation*).

---

## 1. Diagram Alir Sistem Terpadu (End-to-End Architecture)

Berikut adalah diagram alir lengkap dari proses *ingestion* data mentah hingga evaluasi metrik akhir:

```mermaid
flowchart TD
    subgraph S1["1. Data Ingestion & Alignment"]
        A1["ERA5 Reanalysis (2020-2026)\nSuhu, Kelembapan, Tekanan, Dewpoint"]
        A2["Satelit GSMaP NRT (2020-2024)\nCurah Hujan Makro (Pre-Training)"]
        A3["Stasiun AWS Jerukagung (2025-2026)\nGround Truth Mikro (Fine-Tuning)"]
        A1 --> B1["Temporal UTC Resampling & Alignment (1 Jam)"]
        A2 --> B1
        A3 --> B1
    end

    subgraph S2["2. Feature Engineering & Scaling"]
        B1 --> C1["Lag Features (t-1, t-2, t-3 jam)"]
        B1 --> C2["Rolling Moving Average (3h, 6h, 12h, 24h)"]
        B1 --> C3["Cyclic Time Encoding (sin/cos Jam & Bulan)"]
        C1 & C2 & C3 --> C4["Feature Matrix (X)"]
        C4 --> C5["MinMaxScaler (Fit HANYA pada Data Train)"]
    end

    subgraph S3["3. Two-Stage Hurdle Architecture"]
        C5 --> D1["Stage 1: Binary Classification (Occurence)\nAmbang Batas Hujan: P >= 0.2 mm/jam"]
        C5 --> D2["Stage 2: Conditional Amount Regression\nDilatih HANYA pada Sampel Hujan (P >= 0.2 mm)"]
        D1 --> E1["Probabilitas Hujan (p)\nThresholding: p >= 0.5 -> Hujan (1)"]
        D2 --> E2["Estimasi Magnitudo Intensitas (mm/jam)"]
        E1 & E2 --> E3["Output Akhir Terkondisi:\nJika p < 0.5 -> 0.0 mm/jam\nJika p >= 0.5 -> Intensitas Regresi"]
    end

    subgraph S4["4. Dual Model & Transfer Learning"]
        E3 --> F1["Model A: Tree-Based XGBoost\n- Optuna HPO Dynamic Search\n- Warm-Start Fine-Tuning"]
        E3 --> F2["Model B: Deep Learning BiLSTM\n- Input: Sliding Window 24 Jam\n- Layer Freezing & Top Retraining"]
    end

    subgraph S5["5. Comprehensive Evaluation & Inference"]
        F1 & F2 --> G1["Klasifikasi: Precision, Recall, F1-Score, CSI, POD, FAR, ROC-AUC, PR-AUC"]
        F1 & F2 --> G2["Regresi: MAE, RMSE, NRMSE Range %, Rasio RMSE/SD, Bias Presipitasi"]
        F1 & F2 --> G3["Hidrologi: Kurva Akumulasi Curah Hujan Kumulatif (Water Balance)"]
    end
```

---

## 2. Ingestion, Rekayasa Fitur & Distribusi Kelas Data

Data atmosfer dan presipitasi diselaraskan pada resolusi waktu jam-jaman (*hourly*) dengan pembagian domain makro dan mikro.

```mermaid
flowchart LR
    subgraph RawData["Sumber Data Mentah"]
        R1["ERA5 Reanalysis\n- Temperature 2m (°C)\n- Relative Humidity (%)\n- Dewpoint Temp (°C)\n- Surface Pressure (hPa)"]
        R2["Target Makro:\nGSMaP NRT Satellite (mm/h)"]
        R3["Target Mikro:\nAWS IoT Jerukagung (mm/h)"]
    end

    subgraph FE["Ekstraksi Fitur Turunan"]
        F_Lag["Fitur Lag Temporal:\n- T_lag1, T_lag2, T_lag3\n- RH_lag1, RH_lag2, RH_lag3\n- Pres_lag1, Pres_lag2\n- Rain_lag1, Rain_lag2"]
        F_Roll["Statistik Bergerak (Rolling Window):\n- Mean Suhu 3h, 6h, 12h, 24h\n- Mean RH 3h, 6h, 12h, 24h\n- Std RH & Tekanan 6h"]
        F_Cyc["Siklus Diurnal & Musiman:\n- sin_hour, cos_hour (Periode 24h)\n- sin_month, cos_month (Periode 12m)"]
    end

    subgraph OutputFE["Matriks Input Model"]
        X_Out["Matriks Fitur (X)\nTotal Fitur: 28 Kolom Variabel"]
    end

    R1 & R2 & R3 --> F_Lag & F_Roll & F_Cyc
    F_Lag & F_Roll & F_Cyc --> X_Out
```

### Visualisasi Distribusi Kelas (Tidak Hujan vs Hujan):
Karakteristik ketidakseimbangan kelas (*class imbalance*) pada kedua domain pengamatan ditampilkan secara terpisah di bawah ini:

| (A) Domain Pre-Training (Satelit GSMaP) | (B) Domain Fine-Tuning (Stasiun AWS IoT) |
|:---:|:---:|
| ![Distribusi GSMaP](outputs_inference/figures/distribution_rain_pretrain.png) | ![Distribusi AWS](outputs_inference/figures/distribution_rain_finetune.png) |

> **Catatan Pencegahan Kebocoran Data (*Zero Data Leakage Rule*):**
> `MinMaxScaler` diinisialisasi dan di-*fit* **hanya pada data latih (*training set*)**. Data uji (*test set*) dan tahap *fine-tuning* hanya menerapkan fungsi `.transform()` menggunakan parameter *scaler* yang telah dibekukan (`pretrain_scaler.pkl`).

---

## 3. Arsitektur Pemodelan Dua Tahap (*Two-Stage Hurdle Architecture*)

Data curah hujan jam-jaman didominasi oleh nilai nol (*zero-inflated*, $>85\%$ tidak hujan). Pendekatan *single regression* standar sering gagal karena menghasilkan prediksi gerimis semu (*drizzle artifacts*). Untuk mengatasinya, digunakan arsitektur **Two-Stage Hurdle Model**:

```mermaid
flowchart TD
    InputSample["Input Sampel Fitur Jam ke-t (X_t)"] --> Stage1["Stage 1: Binary Classifier\n(P >= 0.2 mm/jam)"]
    InputSample --> Stage2["Stage 2: Amount Regressor\n(Dilatih pada P >= 0.2 mm)"]
    
    Stage1 --> ProbOut["Probabilitas Hujan: P(Hujan)"]
    ProbOut --> Decision{"Apakah P(Hujan) >= 0.5?"}
    
    Decision -- "TIDAK (0)" --> PredZero["Prediksi Akhir: 0.0 mm/jam\n(Kering / Tidak Hujan)"]
    
    Decision -- "YA (1)" --> ApplyReg["Ambil Nilai Magnitudo dari Stage 2"]
    Stage2 --> MagOut["Prediksi Magnitudo Intensitas (mm/jam)"]
    MagOut --> ApplyReg
    ApplyReg --> PredRain["Prediksi Akhir: y_pred = max(0.2, Magnitudo)\n(Hujan Terkondisi)"]
```

---

## 4. Strategi Transfer Learning & Kuantifikasi *Delta Improvement*

Model memanfaatkan data historis jangka panjang satelit ($2020 - 2024$) untuk mempelajari pola makro pembentukan hujan, kemudian diadaptasikan ke stasiun darat AWS ($2025 - 2026$) untuk mengoreksi bias instrumen lokal.

```mermaid
sequenceDiagram
    autonumber
    participant SatData as Dataset Satelit GSMaP (Makro)
    participant Model as Pipeline XGBoost / LSTM
    participant AWSData as Dataset Stasiun AWS (Mikro Ground Truth)
    participant Evaluator as Modul Evaluasi & Inferensi

    Note over SatData,Model: FASE 1: PRE-TRAINING (2020-2024, N ≈ 43.848 Jam)
    SatData->>Model: Latih Model Stage-1 (Klasifikasi) & Stage-2 (Regresi)
    Model->>Evaluator: Uji Generalisasi Makro pada GSMaP Test Split (2024-2025)

    Note over AWSData,Model: FASE 2: FINE-TUNING / DOMAIN ADAPTATION (2025-2026)
    Model->>Model: Muat Checkpoint Pre-Trained Weights
    AWSData->>Model: Latih Lanjut (Low Learning Rate & Layer Freezing)
    
    Note over Model,Evaluator: FASE 3: INFERENSI KOMPARATIF TERPADU
    Model->>Evaluator: Evaluasi 4 Model Sekaligus pada AWS Test Split (2026-04 s.d 2026-05)
    Evaluator->>Evaluator: Hitung Delta Improvement, Kurva ROC/PR, & Curah Hujan Kumulatif
```

### Visualisasi Perbandingan Metrik (Sebelum vs Sesudah Fine-Tuning pada Data AWS):

#### A. Perbandingan Precision, Recall, dan F1-Score (Efek Adaptasi)
![Perbandingan Klasifikasi](outputs_inference/figures/grouped_bar_delta_classification.png)

#### B. Perbandingan MAE, RMSE, dan R² Score (Efek Adaptasi)
![Perbandingan Regresi](outputs_inference/figures/grouped_bar_delta_regression.png)

#### C. Perbandingan ROC-AUC dan PR-AUC (Efek Adaptasi)
![Perbandingan AUC](outputs_inference/figures/grouped_bar_delta_auc.png)

---

### Visualisasi Perbandingan Metrik Antar-Model pada Masing-Masing Tahap:

#### 1. Perbandingan Metrik Klasifikasi Pre-Training (XGBoost vs LSTM pada Satelit GSMaP)
![Klasifikasi Pre-Training](outputs_inference/figures/grouped_bar_pretrain_classification.png)

#### 2. Perbandingan Metrik Regresi Pre-Training (XGBoost vs LSTM pada Satelit GSMaP)
![Regresi Pre-Training](outputs_inference/figures/grouped_bar_pretrain_regression.png)

#### 3. Perbandingan Metrik Klasifikasi Model Fine-Tuned (XGBoost vs LSTM pada Stasiun AWS)
![Klasifikasi Fine-Tuned](outputs_inference/figures/grouped_bar_finetune_classification.png)

---

## 5. Komparasi Deret Waktu (*Time-Series*) & Scatter Plot Korelasi

### A. Perbandingan Time-Series Prediksi Model vs Observasi AWS
Visualisasi di bawah ini menampilkan kesesuaian puncak hidrograf presipitasi antara observasi stasiun AWS dengan hasil prediksi XGBoost (Fine-Tuned) dan LSTM (Fine-Tuned):

![Time-Series Fine-Tuning](outputs_inference/figures/timeseries_comparison_finetune.png)

### B. Perbandingan Sebelum vs Sesudah Fine-Tuning per Model
Memperlihatkan perbaikan signifikan garis prediksi dari tahap awal satelit (merah putus-putus) ke tahap teradaptasi stasiun darat (garis solid):

| (1) Model XGBoost (Pre vs Post Fine-Tuning) | (2) Model LSTM (Pre vs Post Fine-Tuning) |
|:---:|:---:|
| ![FT XGBoost](outputs_inference/figures/finetune_comparison_xgboost.png) | ![FT LSTM](outputs_inference/figures/finetune_comparison_lstm.png) |

### C. Scatter Plot Korelasi Regresi (Sampel Kejadian Hujan)
![Scatter Plot](outputs_inference/figures/scatter_comparison_finetune.png)

---

## 6. Verifikasi Hidrologis: Akumulasi Curah Hujan Kumulatif (*Water Balance*)

Dalam analisis hidrometeorologi dan aplikasi peringatan dini banjir, **konservasi massa volume total presipitasi ($\sum P$)** sepanjang periode pengamatan adalah parameter paling krusial.

![Curah Hujan Kumulatif](outputs_inference/figures/cumulative_rainfall_comparison.png)

> **Interpretasi Fisik:**
> Kurva akumulasi membuktikan bahwa model *Pre-Trained* satelit mengalami *under-estimation* volume total presipitasi. Proses *fine-tuning* berhasil mengoreksi kemiringan kurva akumulasi model ($\text{cumsum}(P_{\text{pred}})$) sehingga berimpit sempurna dengan garis observasi AWS asli ($\text{cumsum}(P_{\text{AWS}})$).

---

## 7. Analisis Matriks Konfusi, Kurva ROC & Precision-Recall

### 7.0 Ilustrasi Konseptual Matriks Konfusi (TP, TN, FP, FN) & Metrik Standar Meteorologi
Diagram referensi di bawah ini menguraikan definisi 4 kuadran klasifikasi biner presipitasi serta formulasi matematis metrik verifikasi standar BMKG/WMO:

![Ilustrasi Konseptual Matriks Konfusi](outputs_inference/figures/ilustrasi_konseptual_confusion_matrix.png)

### 7.1 Matriks Konfusi Empiris Model (Tahap Fine-Tuned pada Stasiun AWS)

| Confusion Matrix: XGBoost (Fine-Tuned) | Confusion Matrix: LSTM (Fine-Tuned) |
|:---:|:---:|
| ![CM XGBoost](outputs_inference/figures/confusion_matrix_xgboost_finetune.png) | ![CM LSTM](outputs_inference/figures/confusion_matrix_lstm_finetune.png) |

### 7.2 Kurva Karakteristik Operasi Penerima (*ROC Curve*) & Precision-Recall (*PR Curve*)

| Kurva ROC Terpadu (4 Model pada AWS) | Kurva Precision-Recall Terpadu (4 Model pada AWS) |
|:---:|:---:|
| ![ROC Terpadu](outputs_inference/figures/roc_curve_comparison_aws.png) | ![PR Terpadu](outputs_inference/figures/pr_curve_comparison_aws.png) |

---

## 8. Ringkasan Ekspor Berkas & Struktur Output

Seluruh keluaran gambar dan data numerik diekspor secara terstruktur ke direktori `outputs_inference/`:

```
outputs_inference/
├── figures/
│   ├── class_distribution_pretrain_vs_finetune.png
│   ├── distribution_rain_pretrain.png
│   ├── distribution_rain_finetune.png
│   ├── grouped_bar_delta_classification.png
│   ├── grouped_bar_delta_regression.png
│   ├── grouped_bar_delta_auc.png
│   ├── grouped_bar_pretrain_classification.png
│   ├── grouped_bar_pretrain_regression.png
│   ├── grouped_bar_finetune_classification.png
│   ├── cumulative_rainfall_comparison.png
│   ├── timeseries_comparison_pretrain.png
│   ├── timeseries_comparison_finetune.png
│   ├── finetune_comparison_xgboost.png
│   ├── finetune_comparison_lstm.png
│   ├── scatter_comparison_pretrain.png
│   ├── scatter_comparison_finetune.png
│   ├── ilustrasi_konseptual_confusion_matrix.png
│   ├── confusion_matrix_xgboost_finetune.png
│   ├── confusion_matrix_lstm_finetune.png
│   ├── roc_curve_comparison_aws.png
│   ├── roc_curve_xgboost.png
│   ├── roc_curve_lstm.png
│   ├── pr_curve_comparison_aws.png
│   ├── pr_curve_xgboost.png
│   └── pr_curve_lstm.png
├── metrics_summary_pretrain.csv / .txt
├── metrics_summary_finetune.csv / .txt
├── validation_objective_rmse_sd_pretrain.csv / .txt
└── validation_objective_rmse_sd_finetune.csv / .txt
```
