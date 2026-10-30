# Arsitektur Two-Stage Hurdle LSTM dengan Bayesian Optimization (Optuna TPE + Hyperband)

**Afiliasi Akademik:** Universitas Putra Bangsa (Kebumen, Jawa Tengah)  
**Lokasi Studi:** AWS IoT Stasiun Klimatologi Jerukagung, Klirong, Kebumen, Jawa Tengah (-7.7667° LS, 109.6167° BT)  
**Tujuan Penelitian:** Prediksi curah hujan ekstrem jangka pendek (*short-term extreme rainfall forecasting*) menggunakan kombinasi data reanalisis atmosfer ERA5 (14 variabel) dan target presipitasi satelit GSMaP/IMERG pada periode 21 tahun (2005–2025).

---

## 1. Struktur Folder `LSTM_Bayesian/`

```text
LSTM_Bayesian/
├── model-lstm-bayesian-optimization.ipynb  # Notebook interaktif dengan 28 visualisasi terpisah (300 DPI)
├── model-lstm-bayesian-optimization.py     # Skrip eksekusi mandiri Python (CLI & pipeline ready)
└── README.md                               # Panduan metodologi dan dokumentasi eksekusi
```

---

## 2. Metodologi: Model Dua Tahap (*Two-Stage Hurdle Model*)

Curah hujan di wilayah tropis Kebumen memiliki sifat *zero-inflated* (>85% jam kering tanpa presipitasi). Model regresi tunggal konvensional gagal menangkap fenomena ini karena menghasilkan *artificial drizzle* konstan. 

Arsitektur Hurdle mengatasi kendala ini melalui dua tahap terpisah:
1. **Tahap 1 (Occurrence Classifier):** Mengklasifikasikan kejadian hujan ($P(\text{hujan}) \ge 0.2\text{ mm/jam}$). Dioptimalkan terhadap metrik **F1-Score** dengan kalibrasi probabilitas Isotonik dan pencarian ambang batas dinamis $\tau \in [0.20, 0.70]$.
2. **Tahap 2 (Intensity Regressor):** Memprediksi intensitas curah hujan dalam transformasi logaritmik $\log(1 + y)$ **hanya pada subset sampel hujan aktual**.
3. **Joint Hurdle Inference:**
   $$\hat{y} = \begin{cases} \exp(\hat{y}_{\text{reg}}) - 1 & \text{jika } \hat{y}_{\text{occ}} = 1 \\ 0 & \text{jika } \hat{y}_{\text{occ}} = 0 \end{cases}$$

---

## 3. Konfigurasi Bayesian Optimization (TPE Multivariat + Hyperband)

- **Multivariate Tree-structured Parzen Estimator (TPE):** Memodelkan kovarians multidimensi antar parameter dengan formula kuantil elite $\gamma = 0.20$ (`gamma=lambda n: min(int(np.ceil(0.20 * n)), 25)`).
- **Hyperband Successive Halving Pruning:** Mengeliminasi model yang tidak menjanjikan pada rung intermediate (epoch 5, 10, 15), menghemat hingga 70% waktu komputasi GPU.
- **Lean Search Space (Bebas Bloat):**
  - `units_l1`: `[64, 128]`
  - `units_l2`: `[32, 64]`
  - `dropout`: `[0.1, 0.2, 0.3]`
  - `batch_size`: `[64, 128, 256]`
  - `lr`: `1e-4` s/d `5e-3` (skala log)
- **Pembagian Temporal Invariant (2005–2025):**
  - **Train:** 2005-01-01 s/d 2020-12-31 (16 tahun, multi-siklus ENSO & IOD)
  - **Val:** 2021-01-01 s/d 2023-12-31 (3 tahun)
  - **Test:** 2024-01-01 s/d 2025-12-31 (2 tahun)

---

## 4. Panduan Eksekusi

### A. Menjalankan Jupyter Notebook
Buka `model-lstm-bayesian-optimization.ipynb` di JupyterLab, VS Code, atau unggah ke Kaggle (pilih Accelerator: **GPU T4 x1**). Seluruh path data satelit akan terdeteksi secara otomatis (`/kaggle/input/**/` maupun direktori lokal).

### B. Menjalankan Skrip Python Mandiri
```bash
# Menjalankan pelatihan penuh (50 trials Bayesian HPO)
python LSTM_Bayesian/model-lstm-bayesian-optimization.py

# Untuk uji coba cepat alur kerja (5 trials, 2 epoch per trial)
# Cukup atur RUN_MODE = "PIPELINE_TEST" di awal konfigurasi global
```

---

## 5. Katalog 28 Visualisasi Terpisah (Resolusi 300 DPI)

Setiap grafik diekspor ke folder `outputs_lstm_bayesian/plots/` secara mandiri (*standalone*), tanpa ada subplot yang digabung, dengan penamaan terstandarisasi:

1. `01_distribusi_data_hujan_pretraining.png` — Distribusi frekuensi kelas Tidak Hujan vs Hujan (2005–2025).
2. `02_training_loss_classifier.png` — Kurva loss Binary Cross-Entropy model klasifikasi terbaik.
3. `03_bayesian_clf_konvergensi_optimasi.png` — Riwayat konvergensi F1-Score sepanjang trial Bayesian Classifier.
4. `04_bayesian_clf_kepentingan_hiperparameter_fanova.png` — Derajat signifikansi hiperparameter classifier (fANOVA).
5. `05_bayesian_clf_response_surface_lr_batch.png` — Lanskap 2D interaksi Learning Rate vs Batch Size.
6. `06_bayesian_clf_efisiensi_pruning_hyperband.png` — Rasio trial lengkap vs dipruning oleh Hyperband.
7. `07_bayesian_clf_densitas_parzen_tpe_lr.png` — Estimasi densitas probabilitas Parzen Kernel $\ell(x)$ vs $g(x)$.
8. `08_bayesian_clf_koordinat_paralel.png` — Koordinat paralel multidimensi lintasan hiperparameter classifier.
9. `09_bayesian_clf_marginal_slice_units_l1.png` — Efek marginal kapasitas neuron LSTM L1 terhadap F1.
10. `10_bayesian_clf_marginal_slice_dropout.png` — Efek marginal tingkat regularisasi Dropout terhadap F1.
11. `11_training_loss_regressor.png` — Kurva loss MSE model regresi intensitas terbaik.
12. `12_bayesian_reg_konvergensi_optimasi.png` — Riwayat konvergensi minimasi MSE model regresi intensitas.
13. `13_bayesian_reg_kepentingan_hiperparameter_fanova.png` — Derajat signifikansi hiperparameter regressor (fANOVA).
14. `14_bayesian_reg_response_surface_units_l1_l2.png` — Lanskap interaksi kapasitas Units L1 vs L2 terhadap MSE.
15. `15_bayesian_reg_efisiensi_pruning_hyperband.png` — Rasio alokasi komputasi pruning Hyperband regressor.
16. `16_bayesian_reg_densitas_parzen_tpe_lr.png` — Estimasi densitas Parzen Kernel $\ell(x)$ vs $g(x)$ untuk regresi.
17. `17_bayesian_reg_koordinat_paralel.png` — Koordinat paralel lintasan hiperparameter regressor.
18. `18_bayesian_reg_marginal_slice_units_l1.png` — Efek marginal kapasitas neuron LSTM L1 terhadap MSE.
19. `19_bayesian_reg_marginal_slice_batch_size.png` — Efek marginal ukuran batch terhadap MSE regresi.
20. `20_confusion_matrix_two_stage.png` — Matriks konfusi klasifikasi Two-Stage Hurdle (Colormap Purples, fmt=',d', 16pt bold).
21. `21_two_stage_timeseries_prediction.png` — Profil runtun waktu prediksi model dua tahap vs observasi ground truth (336 jam).
22. `22_two_stage_roc_curve.png` — Kurva ROC model dua tahap beserta skor AUC.
23. `23_two_stage_precision_recall_curve.png` — Kurva Precision-Recall beserta nilai PR-AUC.
24. `24_two_stage_calibration_curve.png` — Kurva kalibrasi reliabilitas probabilitas model dua tahap.
25. `25_two_stage_prediction_vs_observation.png` — Diagram pencar intensitas prediksi vs observasi sampel hujan aktual.
26. `26_two_stage_residual_distribution.png` — Distribusi residual error prediksi dan estimasi densitas normal.
27. `27_two_stage_cumulative_rainfall.png` — Akumulasi presipitasi kumulatif model dua tahap sepanjang periode uji.
28. `28_shap_feature_importance.png` — Diagram ringkasan nilai Shapley (SHAP) untuk interpretabilitas fitur atmosfer.
