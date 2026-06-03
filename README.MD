# Meteorology Analysis

Repository ini berisi kumpulan notebook dan dataset untuk analisis data meteorologi, khususnya terkait prediksi curah hujan dan analisis cuaca di Kebumen, Indonesia. Proyek ini menggunakan berbagai teknik machine learning dan deep learning untuk eksplorasi, visualisasi, pembersihan data, serta prediksi biner hujan.

## Struktur Folder

- **deep_learning/**  
  Berisi notebook dan dataset terkait eksperimen deep learning, seperti Multi Layer Perceptron (MLP), LSTM, dan object detection pada tanaman tomat.
- **deepnote/**  
  Berisi notebook eksplorasi, analisis, dan pembersihan data, termasuk analisis Firebase, pembuatan dataset, serta notebook utama _Skripsi Rainfall Biner Prediction_.

## Fitur Utama

- Analisis data cuaca harian dan bulanan dari berbagai sumber.
- Pembersihan data (handling missing values, outlier, dsb).
- Visualisasi data cuaca (heatmap, plot, dsb).
- Prediksi curah hujan menggunakan model machine learning (Decision Tree, Random Forest, SVM, KNN, Neural Network).
- Eksperimen deep learning untuk prediksi dan deteksi objek.

## Cara Penggunaan

1. **Kloning repository:**
   ```sh
   git clone https://github.com/evanalif113/Meteorology-Analysis.git
   ```
2. **Buka notebook yang diinginkan** di folder `deepnote/` atau `deep_learning/` menggunakan Jupyter Notebook, Deepnote, atau VS Code.
3. **Pastikan dependensi berikut terpasang:**
   - pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, missingno, dsb.
   - Instalasi:  
     ```sh
     pip install -r requirements.txt
     ```
     (atau instal manual sesuai kebutuhan notebook)

4. **Jalankan sel-sel notebook sesuai urutan untuk melakukan analisis atau pelatihan model.**

## Dataset

- Data cuaca harian dan bulanan (`datacuacaMei.csv`, `datacuacaJuni.csv`, `perceptron.csv`, dsb).
- Data hasil ekspor dari Firebase dan Meteostat.

## Catatan

- Notebook utama untuk prediksi biner hujan: `deepnote/Skripsi Rainfall Biner Prediction.ipynb`
- Beberapa file data besar tidak disertakan di repository, pastikan Anda memiliki akses ke sumber data yang diperlukan.

---

Kontribusi dan saran sangat terbuka untuk pengembangan lebih