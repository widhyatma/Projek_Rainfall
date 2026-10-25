

### Environment-Aware Array Computation (CuPy vs NumPy)
When performing heavy array computations (e.g., sequence extraction, large matrix manipulation) in Jupyter Notebooks:
1. **Dynamic Import**: Detect if the environment is Kaggle (e.g., `os.path.exists('/kaggle/input')`).
2. **Namespace Safety**: **NEVER alias CuPy as `np`** (i.e., do NOT use `import cupy as np`). Doing so breaks Pandas, Scikit-Learn, and other libraries that rely on standard NumPy behaviors and objects (like Pandas `Index` or `Series`).
3. **Execution Module Alias (`xp`)**: Always import standard numpy as `np`. Then, create a dynamic alias `xp` for heavy computations:
   ```python
   import numpy as np
   import os
   
   if os.path.exists('/kaggle/input'):
       try:
           import cupy as xp
           print("Using CuPy (xp) for GPU Arrays")
       except ImportError:
           import numpy as xp
   else:
       import numpy as xp
   ```
4. **Library Compatibility Warning**: When using `xp` (CuPy) to create arrays, remember that libraries like `scikit-learn` (Scalers), `Pandas`, and `TensorFlow`/`Keras` require standard `numpy.ndarray`. Always convert CuPy arrays back to NumPy using `xp.asnumpy(array)` (or `.get()`) BEFORE passing them to these libraries.


### Hyperparameter Optimization (HPO) Strategy
When deciding between Optuna (dynamic HPO) and Hardcode (static architecture) for predictive models in this project:
1. **Tree-Based Models (XGBoost/LightGBM)**: Always integrate **Optuna** (e.g., 30-50 trials) directly into the notebook. These models are fast to train and memory-safe, meaning Optuna yields significant performance gains without risking environment timeouts or OOM (Out-of-Memory) errors.
2. **Deep Learning Models (LSTM/ANN)**: Always use a **Hardcoded** robust baseline architecture inside production/training notebooks (e.g., BiLSTM 64 -> 32, Dropout 0.2). Do NOT use dynamic Optuna loops for Keras/TensorFlow model architecture search within notebooks to prevent GPU memory leaks (	f.keras.backend.clear_session() is often insufficient in long loops) and Kaggle 12-hour timeouts. If deep learning HPO is strictly necessary, it must be delegated to a standalone optimization script, never in the daily execution pipeline.
3. **Hardcode Readability**: When hardcoding deep learning parameters, extract all hyperparameters (Epochs, Batch Size, Learning Rate, Layer Units) to a prominent configuration block at the top of the training cell for easy manual adjustments.


### Hydrometeorological Analysis Folder Architecture & Methodology
When working on precipitation analysis across Google Earth Engine folders (`Google_Earth_Engine/Dokumen Analisis/`):
1. **Analisis_Data_Satelit_dengan_AWS**: Focuses on 1-hour resolution micro-validation against local AWS IoT Jerukagung ground truth (2025-2026) and 20-year climate trends.
2. **Analisis_Data_Presipitasi_Satelit**: Focuses on long-term daily 22-year (2004-2026) inter-comparison across 8 products using CHIRPS_RNL as continuous benchmark.
3. **Analisis Curah Hujan**: Focuses on multi-scale comparisons (1h vs 24h), 24-hour diurnal cycle, and ACF/PACF lookback analysis.
4. **Analisis Tahunan**: Focuses on multi-year aggregate precipitation trends and annual climate indices.



### Sequence Length & Lookback Determination via ACF/PACF Analysis
When configuring sequence length (lookback window) for LSTM/BiLSTM models:
1. **Statistical Dependency Horizon**: Base sequence length on empirical Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) curves:
   - **Primary Memory ($1/e$ Decorrelation)**: Lags where ACF drops below $1/e \approx 0.368$ identify the core temporal autoregression (typically 2-5 hours for rainfall).
   - **Harmonic Diurnal Multiples**: For atmospheric thermodynamic features (Temperature, Relative Humidity, Pressure) showing strong 24-hour diurnal peaks, the sequence length must capture at least 1 full diurnal cycle ($k \times 24$ hours).
2. **Resolution-Specific Lookback Standards**:
   - **Hourly Resolution (DeepLearning_Satelit / DeepLearning_Meteorologi)**: Standardize on `LOOKBACK = 24` (24 hours / 1 full diurnal day).
   - **1-Minute Micro-Resolution (Model_Imputasi)**: Standardize on `LOOKBACK = 120` (2 hours) complemented by explicit cyclic time features (`sin_hour`, `cos_hour`, `sin_day`).
   - **Daily Resolution (Analisis_Data_Presipitasi_Satelit)**: Standardize on `LOOKBACK = 7 to 14` (synoptic atmospheric wave timescale).
3. **Efficiency Guardrail**: Avoid arbitrary large lookback windows (>72h for hourly data) unless multiday synoptic memory is empirically validated, as excessive sequence lengths dilute gradient signals and increase GPU VRAM without improving precipitation onset skill.


### Scientific Confusion Matrix & Visualization Standards
When rendering Confusion Matrices for model evaluation and conceptual verification in this repository:
1. **Model Evaluation Matrices (XGBoost vs LSTM)**:
   - **Clean Numeric Format**: Display raw sample counts ($N$) with comma thousand separators (`fmt=',d'`). Never add verbose cards, subtitles, or redundant formula annotations inside evaluation heatmaps.
   - **Model-Differentiated Colormaps**: 
     - **Tree-Based (XGBoost)**: Standardize on `cmap='Blues'`.
     - **Deep Learning (LSTM / BiLSTM)**: Standardize on `cmap='Purples'`.
   - **Auto-Luminance Contrast & Cell Borders**: Never hardcode dark font colors in `annot_kws`. Ensure dark cells render with bold white text and light cells with bold dark text. Separate cells with clean white grid lines (`linewidths=2.0, linecolor='white'`).
   - **Typography & Proportions**: Use bold 16pt annotations inside cells, 11-12pt bold for axis/tick labels, and 13pt bold for titles.
2. **Conceptual/Pedagogical Example Matrices**:
   - For methodology illustrations or journal theory explanations, create dedicated 2x2 plots containing **ONLY** the 4 quadrant acronyms: `TN` (top-left), `FP` (top-right), `FN` (bottom-left), and `TP` (bottom-right) centered in large bold font.


### High-Resolution Universal Meteorological Imputation Architecture (`Model_Imputasi`)
When handling missing data reconstruction across AWS IoT meteorological ground stations (`id-02`, `id-03`, `id-05` at 1-minute resolution):
1. **Two-Phase Hybrid Imputation Strategy**:
   - **Phase 1 (Micro-Gaps $\le 15$ minutes)**: Apply Linear / Spline Interpolation for high computational efficiency and local continuity.
   - **Phase 2 (Macro-Gaps $> 15$ minutes up to multi-day outages)**: Apply a Multi-Input Denoising BiLSTM Autoencoder conditioned on interpolated ERA5 macro-meteorological reanalysis and cyclical temporal features ($\sin/\cos$ hour, day, month).
2. **Multi-Input Network Topology**:
   - **Continuous Stream**: Shape `(SEQ_LEN=120, num_features)` containing sensor values ($T, RH, P, T_{\text{dew}}$), ERA5 anchors, cyclic time encodings, and binary observation masks.
   - **Categorical Stream**: Node ID mapped via `layers.Embedding(num_nodes, embedding_dim=4)` and expanded via `RepeatVector(120)` to allow a single universal neural network to specialize across heterogeneous microclimates.
   - **Layer Architecture**: Hardcoded `Bidirectional(LSTM(64))` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Bidirectional(LSTM(32))` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `TimeDistributed(Dense(4))`.
3. **Loss Function & Training Protocol**:
   - Supervised Denoising with Mask-Weighted Mean Squared Error (`sample_weight = mask`) so gradient updates only backpropagate through ground truth observations (`tf.math.is_finite(y_true)`).
   - Multi-rate synthetic missingness validation (10% to 50% simulated dropout) evaluated using MAE, RMSE, $R^2$, NSE (Nash-Sutcliffe), and KGE (Kling-Gupta).
4. **Physics-Informed Thermodynamic Boundary & De-Jittering Smoothing**:
   - **Magnus-Tetens Consistency**: Always enforce $T_{\text{dew}} \le T_{\text{ambient}}$ across all reconstructed timesteps (`df['dewpoint'] = np.minimum(df['dewpoint'], df['temperature'])`).
   - **Physical Humidity Bounding**: Hard clip relative humidity to $10\% \le RH \le 100\%$.
   - **Savitzky-Golay Zero-Phase Filtering**: Apply Savitzky-Golay filter ($W=15, \text{poly}=2$) to eliminate sensor ADC quantization stepping while preserving diurnal crests, troughs, and rain-induced drops.


### SQLite Local Telemetry Cache & Direct Query Protocol (`meteo_local_cache.db`)
When sourcing training or validation telemetry from the SQLite cache at `D:\Github\Firebase_Database_Administrator\databases\meteo_local_cache.db`:
1. **Schema & Table**: Query table `aws_measurements` with columns `station_id`, `unix_ts`, `temperature`, `humidity`, `pressure`, `dew_point`, `dew`, `volt`.
2. **Station Filtering**: Restrict training to calibrated, long-term stable outdoor nodes (`id-03` and `id-05`). Omit unstable nodes (such as `id-01`) until telemetry stabilizes.
3. **Column Deduplication & Dewpoint Unification**:
   - Always unify `dew_point` and `dew` into a single `dewpoint` column before DataFrame indexation:
     ```python
     if 'dew_point' in df.columns and 'dew' in df.columns:
         df['dewpoint'] = df['dew_point'].combine_first(df['dew'])
         df = df.drop(columns=['dew_point', 'dew'])
     ```
   - Ensure all columns are deduplicated (`df = df.loc[:, ~df.columns.duplicated()]`) before applying Quality Control bounds to avoid 2D boolean indexer errors.
4. **Datetime Parsing & Resampling**:
   - Convert unix seconds to Jakarta local time: `pd.to_datetime(df['unix_ts'], unit='s', utc=True).dt.tz_convert('Asia/Jakarta').dt.tz_localize(None)`.
   - Drop duplicate timestamps, set datetime index, and resample to continuous 1-minute grid (`df.resample('1min').mean()`).
5. **Standard 10-Digit UNIX Epoch Generation**:
   - When generating timestamp columns for export or database synchronization, ensure exact 10-digit integer seconds:
     ```python
     df['timestamp'] = ((pd.to_datetime(df.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') - 7 * 3600).astype(int)
     ```


### Firebase Realtime Database Telemetry Synchronization & Conflict Avoidance
When uploading reconstructed or smoothed time-series records to Firebase Realtime Database:
1. **Sub-Minute Record Purge**: Prior to writing regular 1-minute time series, inspect and delete conflicting sub-minute / non-integer minute timestamps (e.g. `:04`, `:18`, `:35` seconds from indoor testing) to prevent dashboard charts from displaying high-frequency square-wave / zig-zag oscillations.
2. **Integer Minute Normalization**: Ensure every synchronized record sits on a pure `:00` second boundary matching standard AWS meteorological telemetry.


### Thesis Academic Framework & Scientific Narrative (Jerukagung Case Study)
When discussing, documenting, or reporting results for the short-term rainfall prediction models:
1. **Core Thesis Title & Scope**:
   - Title: *"Analisis Komparatif Performa Model XGBoost dan LSTM dalam Prediksi Curah Hujan Jangka Pendek (Studi Kasus: Desa Jerukagung, Klirong, Kebumen)"*.
   - Location: Desa Jerukagung, Kecamatan Klirong, Kabupaten Kebumen, Jawa Tengah (AWS IoT Stasiun Klimatologi Jerukagung).
2. **Research Problem & Justification**:
   - Zero-Inflation Challenge: Rainfall in coastal/lowland Kebumen exhibits >85% zero precipitation hours; standard regression yields artificial drizzle. The Two-Stage Hurdle architecture is the primary methodological solution.
   - Spatial Mismatch Challenge: Satellite products (GSMaP, IMERG at ~11 km grid) and reanalysis (ERA5 at ~31 km grid) lack micro-scale convective representation, necessitating domain adaptation (fine-tuning) to local IoT ground truth.
3. **Model Comparison Framing (XGBoost vs LSTM)**:
   - **XGBoost (Tree-Based)**: Evaluated as a high-precision, computationally lightweight benchmark with Bayesian Optuna HPO; excels in precision and lower overall RMSE.
   - **LSTM (Deep Learning)**: Evaluated as a sequential diurnal memory network (24-hour sliding window); excels in temporal dependency tracking, higher recall, and detecting rain onset transitions.
   - **Scientific Verdict**: Both models demonstrate dramatic performance surges post-fine-tuning (Precision +42%, R² shifting from negative to +0.50), proving that satellite-to-ground domain adaptation is viable for localized hydrometeorological forecasting in data-sparse tropical environments.


### Google Earth Engine (GEE) Temporal Boundary & Satellite Dataset Standards (2005–2025)
When writing, modifying, or executing Google Earth Engine (GEE) JavaScript scripts and preparing satellite tabular time-series for the Jerukagung rainfall research:
1. **Core Observation Window (2005–2025)**:
   - Standardize all satellite pre-training, climatological trend analysis, and benchmark inter-comparisons (ERA5 Hourly, JAXA GSMaP, NASA GPM IMERG) to the **21-year period: 1 January 2005 through 31 December 2025**.
2. **GEE End-Exclusive Filtering Guardrail**:
   - Because GEE `ee.ImageCollection.filterDate(start, end)` applies a semi-open interval `[start, end)`, specifying `'2025-12-31'` erroneously truncates all hours of December 31st at `00:00:00 UTC`.
   - Always specify the end date as the first day of the following year:
     ```javascript
     .filterDate('2005-01-01', '2026-01-01')
     ```
   - This ensures complete, uninterrupted coverage up to `2025-12-31 23:00:00 UTC` (or `23:30:00 UTC` for IMERG).
3. **Theoretical Row Verification Standards**:
   - **Hourly Datasets (ERA5, GSMaP)**: Exactly 21 years with 5 leap years (2008, 2012, 2016, 2020, 2024) = 7,670 days * 24 = 184,080 rows.
   - **Half-Hourly Datasets (IMERG)**: Exactly 184,080 * 2 = 368,160 rows.
   - Sanity checks must confirm `min(datetime) == '2005-01-01 00:00:00'` and `max(datetime) == '2025-12-31 23:00:00'` with 0 missing temporal steps before saving to `Data_Satelit/`.
4. **Fine-Tuning & Domain Adaptation Temporal Bridge**:
   - Ground station observations (AWS IoT `id-05_clear_data_hourly.csv`) span `2025-01-01` to `2026-05-31` (or later).
   - When preparing ERA5 for fine-tuning, retain the 2026 observation bridge or construct the fine-tuning temporal index directly from station observation timestamps to avoid empty validation/test splits in 2026.
5. **Historical Model Pre-Training Split Invariant (1998–2025)**:
   - For neural network pre-training and Two-Stage modeling in `DeepLearning_Satelit/`, the historical split is strictly fixed and must NEVER be altered:
     - `TRAIN_START = '1998-01-01'` to `TRAIN_END = '2020-12-31'`
     - `VAL_START = '2021-01-01'` to `VAL_END = '2023-12-31'`
     - `TEST_START = '2024-01-01'` to `TEST_END = '2025-12-31'`
   - Retain both `TRAIN_*` and `PRETRAIN__*` variable identifiers for total backward compatibility across notebooks.


### Academic Research Affiliation & Citation Standards
When generating, updating, or authoring metadata, titles, headers, LaTeX manuscripts, or documentation for research studies in this repository:
1. **Mandatory Institutional Affiliation**:
   - Always state the author affiliation as **Universitas Putra Bangsa** (Kebumen, Jawa Tengah).
   - Never infer, hallucinate, or substitute any other university or institution unless explicitly instructed by the user.
2. **Study Area & Thesis Context**:
   - Location: Desa Jerukagung, Kecamatan Klirong, Kabupaten Kebumen, Jawa Tengah (AWS IoT Stasiun Klimatologi Jerukagung).


### Deep Learning Bayesian Optimization (Optuna TPE Multivariate + Hyperband) Architecture
When designing or extending Bayesian Optimization pipelines for Deep Learning models (LSTM/BiLSTM):
1. **Multivariate TPE Sampler**:
   - Always configure `TPESampler(multivariate=True, n_startup_trials=15, gamma=lambda n: min(int(np.ceil(0.20 * n)), 25), warn_independent_sampling=False)` to model multidimensional covariance between dependent parameters (e.g., learning rate vs batch size, layer units vs regularizers).
   - **Callable Gamma Guardrail**: Never pass a raw float (e.g. `gamma=0.20`) to `TPESampler`. Optuna evaluates `self._gamma(n)` dynamically as a function that takes trial count $N$ and returns the integer count of elite trials for $\ell(x)$. Passing a raw float triggers `TypeError: 'float' object is not callable`.
2. **Successive Halving Pruning Protocol**:
   - Use `HyperbandPruner(min_resource=5, max_resource=30, reduction_factor=3)` to eliminate poor configurations at intermediate rungs (epochs 5, 10, 15), saving 60-75% of GPU compute time.
3. **Zero-Dependency Optuna-Keras Pruning Callback**:
   - In Optuna 4+, never rely on `optuna.integration.TFKerasPruningCallback` (which fails due to decoupled sub-packages). Always implement a native custom callback:
     ```python
     class OptunaKerasPruningCallback(tf.keras.callbacks.Callback):
         def __init__(self, trial, monitor='val_loss'):
             super().__init__()
             self.trial = trial
             self.monitor = monitor

         def on_epoch_end(self, epoch, logs=None):
             logs = logs or {}
             val = logs.get(self.monitor)
             if val is not None:
                 self.trial.report(float(val), step=epoch)
                 if self.trial.should_prune():
                     raise optuna.TrialPruned(f"Pruned at epoch {epoch}")
     ```
4. **Regularization & Two-Stage Hurdle Search Spaces**:
   - Include L2 `weight_decay` kernel regularization ($10^{-6}$ to $10^{-3}$, log scale) alongside dropout to prevent memory gate saturation.
   - For classification: Optimize validation F1-Score via dynamic threshold search ($\tau \in [0.20, 0.70]$) to counteract zero-inflation (>85% dry hours).
   - For regression: Optimize log-transformed precipitation ($\log(1+y)$) MSE on positive rain occurrences.
5. **Mandatory Diagnostic & Publication Visualizations**:
   - Every Bayesian HPO study must export high-resolution (300 DPI) static figures to `outputs_*/plots/`:
     - Panel 1: Optimization History with vertical startup transition line ($N=15$) and cumulative best step curve.
     - Panel 2: Hyperparameter Importances via fANOVA (`optuna.importance.get_param_importances`).
     - Panel 3: 2D Response Surface / Parameter Interaction Landscape.
     - Panel 4: Pruning efficiency & resource allocation breakdown.
   - Save full study trials to CSV (`optuna_trials_*.csv`) and SQLite database for full experimental provenance.

