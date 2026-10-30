# Walkthrough: Universal Bi-LSTM Imputation Model Training (Node 3 & Node 5)

Pelatihan model Universal Bi-LSTM Imputation Network telah berhasil diselesaikan menggunakan database SQLite lokal (`meteo_local_cache.db`) khusus untuk **Node 3 (`id-03`)** dan **Node 5 (`id-05`)** yang telah terbukti stabil.

---

## 1. Ringkasan Data yang Digunakan

Dari database SQLite `D:\Github\Firebase_Database_Administrator\databases\meteo_local_cache.db`:
- **Node 3 (`id-03`)**: 1.418.081 baris data 1-menitan (periode 11 Des 2023 – 22 Agt 2026).
- **Node 5 (`id-05`)**: 750.101 baris data 1-menitan (periode 19 Mar 2025 – 22 Agt 2026).
- **ERA5 Macro Anchors**: 1.850.341 baris data reanalysis 1-menitan hasil interpolasi.

---

## 2. Arsitektur Model & Parameter Pelatihan

| Parameter | Konfigurasi | Keterangan |
| :--- | :--- | :--- |
| **Lookback Sequence** | $120$ menit (2 jam) | Sesuai aturan lookback data mikroklimat 1-menit |
| **Continuous Features (11)** | $T, RH, P, T_{\text{dew}}$ + $\sin/\cos$ jam, hari + ERA5 | Input kontinu dinormalisasi MinMax |
| **Categorical Input (1)** | Node ID $\rightarrow$ `Embedding(8, 4)` | Embedding spasial multi-stasiun |
| **Backbone RNN** | `BiLSTM(64)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `BiLSTM(32)` $\rightarrow$ `Dropout(0.2)` | Topology hardcoded anti-OOM |
| **Loss Function** | `Mask-Weighted MSE` | Hanya menghitung error pada observasi valid |
| **Total Parameters** | $82.468$ trainable parameters | Ringan, cepat, dan stabil |

---

## 3. Hasil & Konvergensi Loss

Pelatihan menunjukkan kurva konvergensi yang sangat tajam dan mulus tanpa overfitting:

### History Metrik per Epoch:
| Epoch | Train Loss | Train MAE | Val Loss | Val MAE | Status Model Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.0362$ | $0.1300$ | $0.00572$ | $0.0391$ | Model disimpan (`val_loss: 0.00572`) |
| **2** | $0.0115$ | $0.0786$ | $0.00425$ | $0.0286$ | Model disimpan (`val_loss: 0.00425`) |
| **3** | $0.0089$ | $0.0683$ | $0.00315$ | $0.0233$ | Model disimpan (`val_loss: 0.00315`) |
| **4** | $0.0072$ | $0.0614$ | $0.00246$ | $0.0199$ | Model disimpan (`val_loss: 0.00246`) |
| **5** | $0.0061$ | $0.0561$ | $0.00204$ | $0.0163$ | Model disimpan (`val_loss: 0.00204`) |
| **6** | $0.0054$ | $0.0522$ | **0.00185** | **0.0159** | **Best Weights Restored (`val_loss: 0.00185`)** |

---

## 4. Artefak Output Model

File-file model dan metadata tersimpan lengkap di folder `Model_Imputasi/outputs/`:
- Model Neural Network: [`outputs/model/universal_bilstm.keras`](file:///d:/Github/Projek_Rainfall/Model_Imputasi/outputs/model/universal_bilstm.keras)
- Feature Scalers: [`outputs/model/scalers.joblib`](file:///d:/Github/Projek_Rainfall/Model_Imputasi/outputs/model/scalers.joblib)
- Node Mapping Dictionary: [`outputs/model/node_map.joblib`](file:///d:/Github/Projek_Rainfall/Model_Imputasi/outputs/model/node_map.joblib)
- Kurva Pelatihan: [`outputs/plots/loss_curve.png`](file:///d:/Github/Projek_Rainfall/Model_Imputasi/outputs/plots/loss_curve.png)
- Riwayat Loss: [`outputs/metrics/training_history.csv`](file:///d:/Github/Projek_Rainfall/Model_Imputasi/outputs/metrics/training_history.csv)

---

## 5. Validasi Eksekusi Pipeline

Pipeline inferensi [`impute_pipeline.py`](file:///d:/Github/Projek_Rainfall/Model_Imputasi/impute_pipeline.py) telah diuji coba dan berhasil melakukan rekonstruksi end-to-end dengan menghasilkan dataset mulus berformat unix timestamp 10-digit standar:

```bash
python Model_Imputasi/impute_pipeline.py --input Model_Imputasi/outputs_rekonstruksi/id03_smoothed_physics_20260902.csv --node id-03 --output Model_Imputasi/outputs/imputed_data/test_impute_id03.csv
```
Output validasi:
```
[INFO] Successfully loaded Universal Bi-LSTM model.
[INFO] Applying Physics-Informed Post-Processing & Smoothing...
[SUCCESS] Clean reconstructed dataset saved to Model_Imputasi/outputs/imputed_data/test_impute_id03.csv
```
