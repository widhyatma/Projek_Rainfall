

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
When working on precipitation analysis across Google Earth Engine folders:
1. **Analisis_Data_Satelit_dengan_AWS**: Focuses on 1-hour resolution micro-validation against local AWS IoT Jerukagung ground truth (2025-2026) and 20-year climate trends.
2. **Analisis_Data_Presipitasi_Satelit**: Focuses on long-term daily 22-year (2004-2026) inter-comparison across 8 products using CHIRPS_RNL as continuous benchmark.
3. **Analisis Curah Hujan**: Focuses on multi-scale comparisons (1h vs 24h) and 24-hour diurnal cycle analysis.
