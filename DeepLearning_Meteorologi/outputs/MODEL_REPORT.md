# Operational Rainfall Forecasting Model Report

## 1. Dataset Summary
The pipeline processes meteorological variables from the Open-Meteo archive for the Jerukagung region. 
- **Temporal Resolution:** Interpolated to hourly, then resampled/aggregated to **3-hourly** intervals.
- **Forecast Horizon:** Model predictions correspond to 3-hour accumulative rainfall in the future (t+1 index).
- **Target:** `target_rain_mm` (Continuous) and `target_class` (Categorical, 4 classes).

## 2. Feature Engineering
The pipeline uses multiple predictors:
- **Meteorological Variables:** `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `dewpoint_depression`.
- **Wind Vectors:** Transformed `wind_speed_10m` and `wind_direction_10m` into continuous `wind_u` and `wind_v` components to avoid angular discontinuity at 360 degrees.
- **Hydrological Lags:** Past rainfall variables at t-1, t-2, t-3, t-6, t-12, and t-24.
- **Rolling Windows:** `rain_roll_mean` and `rain_roll_max` for 3h, 6h, 12h, and 24h.
- **Temporal Features:** Cyclical hour (`hour_sin`, `hour_cos`) and day-of-year (`doy_sin`, `doy_cos`) to capture diurnal and seasonal patterns effectively.

## 3. Rainfall Class Definition (BMKG & WMO Based)
Classification is adapted from BMKG hourly guidelines, extrapolated proportionally for 3-hour convective rainfall typical of tropical Indonesia.
- **Class 0 (No Rain):** 0 mm
- **Class 1 (Light Rain):** 0.1 – 15 mm / 3h
- **Class 2 (Moderate Rain):** 15.1 – 30 mm / 3h
- **Class 3 (Heavy Rain):** > 30 mm / 3h

*Reference:* Based on BMKG's hourly thresholds (1-5mm, 5-10mm, >10mm) converted for 3-hour accumulations often utilized in hydrometeorological research (e.g., *Balai Besar MKG* guidelines).

## 4. Multi-Task Architecture
To produce both exact rainfall estimations and probabilistic class memberships without creating conflicting latent spaces:
- **XGBoost Pipeline:** Uses paired independent estimators: `XGBRegressor` (`reg:squarederror`) and `XGBClassifier` (`multi:softprob`).
- **LSTM Pipeline:** Genuine Multi-Task Neural Network. Shared LSTM layers split into two dense heads:
  - Regression Head: Linear activation optimized by Mean Squared Error (MSE).
  - Classification Head: Softmax activation optimized by Categorical Focal Loss to heavily penalize misclassification of rare extreme events.

## 5. Handling Imbalanced Rainfall
- **LSTM:** Utilizes *Categorical Focal Loss* (Gamma=2.0) to focus training gradients on hard-to-predict extreme rainfall (Class 3) rather than the dominating non-rain instances (Class 0).
- **XGBoost:** Utilizes Scikit-learn's `compute_sample_weight('balanced')` to natively penalize the gradient boosting algorithm proportionally to class infrequency.

## 6. Ensemble Stacking
An operational Ensemble layer acts as a Meta-Model.
- Extracts prediction probabilities and regression values from both LSTM and XGBoost on the Validation Set.
- Trains a `LogisticRegression` for final Class mapping.
- Trains a `Ridge` regressor for final Rainfall amount (mm) mapping.
- Provides superior resilience against individual model overfitting.

## 7. Metrics & Operational API Use
The pipeline exports predictions into `outputs/<model>/predictions/operational_forecast.csv` containing:
`timestamp, predicted_rainfall_mm, rain_probability, rain_class, model`

This JSON-ready tabular format guarantees straightforward deployment via REST APIs, fulfilling the requirement for a fully operational end-to-end meteorological pipeline.
