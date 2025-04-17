# Laporan Proyek Machine Learning - Aditiya Saputra (Predictive Analytics untuk Prediksi Harga Saham BBRI (LQ45))

## Domain Proyek

Prediksi harga saham merupakan aspek fundamental dalam analisis pasar keuangan, khususnya dalam investasi dan manajemen risiko, di mana kemampuan untuk memperkirakan pergerakan harga secara akurat dapat meningkatkan efisiensi pengambilan keputusan, meminimalisir risiko, dan memaksimalkan keuntungan. Namun, fluktuasi harga secara real-time akibat kondisi pasar yang berubah-ubah, gangguan eksternal, serta faktor makroekonomi dan psikologis, menjadi tantangan bagi metode prediksi tradisional.

Metode prediksi harga saham konvensional seperti analisis teknikal dan model statistik klasik (misalnya ARIMA) sering kali bergantung pada asumsi linieritas dan kestasioneran data. Model-model ini cenderung kesulitan dalam menangani perilaku pasar yang dinamis dan kompleks. Selain itu, keberadaan noise pasar, seperti spekulasi mendadak atau berita yang memicu volatilitas, memperburuk performa model tradisional dibandingkan dengan pendekatan berbasis pembelajaran mesin [[1]](https://journal.walisongo.ac.id/index.php/square/article/view/5626). Noise dan ketidakpastian ini adalah bagian yang tak terelakkan dari lingkungan pasar nyata.

Oleh karena itu, penggunaan model berbasis machine learning seperti Long Short-Term Memory (LSTM) menjadi lebih fleksibel dalam menghadapi fluktuasi real-time harga saham BBRI. Model ini mampu mengenali pola non-linier dan hubungan jangka panjang dalam data historis, sehingga memberikan prediksi yang lebih adaptif terhadap dinamika pasar yang terus berubah.

## Business Understanding
### Problem Statement
* Bagaimana memprediksi harga penutupan saham BBRI menggunakan data historis?
* Apakah model LSTM mampu menangkap pola jangka panjang dan memberikan prediksi yang akurat untuk saham BBRI?
* Bagaimana performa prediksi model LSTM terhadap data pengujian?

### Objectives
* Membangun model prediksi berbasis LSTM untuk memproyeksikan harga saham BBRI.
* Melakukan preprocessing data harga saham untuk kebutuhan model time series.
* Mengevaluasi performa model dengan metrik akurasi regresi (MAE, RMSE dan R²-Score).

### Solution
Solusi dilakukan dengan pendekatan sebagai berikut:
* Mengambil data historis harga saham BBRI dari tahun 2016 hingga 2024.
* Melakukan preprocessing data, termasuk normalisasi menggunakan MinMaxScaler dan pembentukan sliding window sepanjang 60 hari untuk membentuk sekuens input LSTM.
* Membangun model LSTM dengan arsitektur sederhana dan melakukan pelatihan menggunakan data yang telah dibagi menjadi data training dan testing.
* Mengevaluasi model menggunakan metrik Mean Absolute Error (MAE), Mean Squared Error (MSE) dan R²-Score.
* Membandingkan nilai prediksi dan harga aktual pada data testing.

## Data Understanding
Sumber Data: Dataset diperoleh dari [[IDX]](https://github.com/wildangunawan/Dataset-Saham-IDX/blob/master/Saham/LQ45/BBRI.csv) dengan kode saham BBRI, yaitu saham dari perusahaan Bank Rakyat Indonesia, yang bergerak di sektor perbankan. Data diambil untuk periode 2019-07-29 hingga 2025-02-21, mencakup lebih dari 1.356 baris data harian perdagangan saham.

Struktur Dataset: Dataset ini memiliki beberapa kolom utama yang merepresentasikan informasi harga dan volume perdagangan saham pada setiap harinya. 
Berikut penjelasan tiap kolom:
| Kolom                  | Tipe Data | Deskripsi                                                                 |
|------------------------|-----------|---------------------------------------------------------------------------|
| `date`                 | object    | Tanggal transaksi saham                                                   |
| `previous`             | float64   | Harga penutupan hari sebelumnya                                          |
| `open_price`           | float64   | Harga pembukaan hari ini                                                 |
| `first_trade`          | float64   | Harga transaksi pertama hari ini                                         |
| `high`                 | float64   | Harga tertinggi hari ini                                                 |
| `low`                  | float64   | Harga terendah hari ini                                                  |
| `close`                | float64   | Harga penutupan hari ini                                                 |
| `change`               | float64   | Selisih harga penutupan hari ini dengan hari sebelumnya                  |
| `volume`               | float64   | Volume transaksi dalam satuan lot                                        |
| `value`                | float64   | Nilai transaksi dalam satuan Rupiah                                      |
| `frequency`            | float64   | Frekuensi atau jumlah transaksi                                          |
| `index_individual`     | float64   | Nilai indeks individual saham (jika tersedia)                            |
| `offer`                | float64   | Harga penawaran tertinggi (ask) saat penutupan                           |
| `offer_volume`         | float64   | Volume pada harga penawaran tertinggi                                    |
| `bid`                  | float64   | Harga permintaan tertinggi (bid) saat penutupan                          |
| `bid_volume`           | float64   | Volume pada harga permintaan tertinggi                                   |
| `listed_shares`        | float64   | Jumlah total saham yang tercatat                                         |
| `tradeble_shares`      | float64   | Jumlah saham yang dapat diperdagangkan (free float)                      |
| `weight_for_index`     | float64   | Bobot saham terhadap indeks (jika ada)                                   |
| `foreign_sell`         | float64   | Jumlah penjualan saham oleh investor asing                               |
| `foreign_buy`          | float64   | Jumlah pembelian saham oleh investor asing                               |
| `delisting_date`       | float64   | Tanggal delisting (tidak relevan, seluruh baris bernilai NaN)            |
| `non_regular_volume`   | float64   | Volume transaksi non-reguler (tender offer, crossing)                    |
| `non_regular_value`    | float64   | Nilai transaksi non-reguler                                              |
| `non_regular_frequency`| float64   | Frekuensi transaksi non-reguler                                          |

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses data preparation untuk memastikan data siap digunakan dalam model LSTM. Setiap langkah disusun secara berurutan dengan tujuan meningkatkan kualitas dan relevansi data terhadap kebutuhan model time series.
1. Pemeriksaan Missing Value Untuk mengetahu kolom mana yang terdapat data NULL
    * Dalam proses ini, kita akan menangani nilai hilang yang terdapat dalam dataset. Penanganan nilai hilang dilakukan dengan mengisi nilai-nilai yang hilang menggunakan median dari kolom yang bersangkutan. Hal ini  bertujuan      untuk menjaga integritas data dan memastikan bahwa model yang dibangun tidak terpengaruh oleh nilai yang hilang.
2. Pemilihan Fitur
    * Dari dataset yang tersedia, dipilih empat kolom numerik yang relevan, yaitu: open_price, high, low, dan close
3. Normalisasi Data
    * Data numerik dinormalisasi menggunakan teknik Min-Max Scaling, yang mengubah skala nilai fitur menjadi rentang [0, 1]. Proses ini dilakukan menggunakan MinMaxScaler dari pustaka scikit-learn.
4. Pembagian Data Latih dan Uji
    * Setelah pemilihan Fitur dan Normalisasi Data, dataset di split dengan ratio 80:20, yaitu 80% untuk training dan 20% untuk testing.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan prediksi harga saham harian BBRI (Bank Rakyat Indonesia). Model yang digunakan adalah algoritma Long Short-Term Memory (LSTM) yang termasuk dalam keluarga Recurrent Neural Network (RNN). Algoritma ini dipilih karena memiliki kemampuan dalam mempelajari pola jangka panjang dan menangani data time series secara efektif.
Tahapan Pemodelan : 
1. Membangun Model Sequential Model dibangun menggunakan Sequential() dari Keras
2. Arsitektur Model
   * 4 Lapisan LSTM, masing-masing berisi 50 unit neuron.
   * Dropout sebesar 0.2 diterapkan setelah setiap layer LSTM untuk regularisasi.
   * Output Layer menggunakan Dense(units=4) untuk memprediksi 4 fitur sekaligus: open_price, high, low, dan close.
3. Kompilasi Model
   * Optimizer: adam, digunakan karena mampu menyesuaikan laju pembelajaran secara adaptif.
   * Loss Function: mean_squared_error, sesuai untuk permasalahan regresi.
   * Metrics: mean_absolute_error, untuk mengevaluasi performa model secara lebih interpretatif.
4. Pelatihan Model
   * Data pelatihan dilakukan sebanyak 50 epoch dengan batch size 32.
   * Model divalidasi menggunakan data test yang telah disiapkan sebelumnya.
   * Proses pelatihan dilakukan dengan fit() menggunakan data X_train dan y_train.
Kelebihan dan Kekurangan Algoritma LSTM :

| Kelebihan                                             | Kekurangan                                          |
|-------------------------------------------------------|-----------------------------------------------------|
| Mampu mempelajari pola jangka panjang                | Butuh waktu pelatihan lebih lama                   |
| Cocok untuk data sekuensial seperti time series      | Cenderung kompleks dan sulit diinterpretasikan     |
| Menghindari masalah *vanishing gradient*             | Hyperparameter tuning memerlukan eksperimen        |

### Proses Improvement (Tuning)
Karena hanya satu algoritma yang digunakan (LSTM), maka dilakukan upaya improvement sebagai berikut:
* Menambahkan jumlah layer LSTM menjadi 4 lapisan untuk menangkap kompleksitas pola data saham.
* Menggunakan Dropout 0.2 di tiap layer untuk menghindari overfitting akibat model terlalu kompleks.
* Menyesuaikan arsitektur output (Dense(units=4)) agar model dapat memprediksi seluruh fitur harga sekaligus.
  
Upaya ini bertujuan untuk meningkatkan kemampuan generalisasi model terhadap data baru sekaligus menjaga performa pada data pelatihan.

## Evaluation
Pada tahap evaluasi model, tujuan utama adalah untuk memahami seberapa baik model LSTM yang dibangun dapat memprediksi harga penutupan saham BBRI berdasarkan data historis. Untuk menilai performa model, kami menggunakan beberapa metrik evaluasi yang sesuai dengan jenis masalah regresi, yaitu **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, dan **R-squared (R²) Score**.

#### Metrik Evaluasi yang Digunakan

1. **Mean Absolute Error (MAE)**:
   - MAE mengukur rata-rata perbedaan absolut antara nilai yang diprediksi oleh model dan nilai yang sebenarnya. Metrik ini memberikan gambaran yang jelas tentang kesalahan prediksi model dalam satuan yang sama dengan data asli.
   
2. **Root Mean Squared Error (RMSE)**:
   - RMSE mengukur kesalahan prediksi model dalam satuan asli dan memberikan penalti lebih besar pada kesalahan yang lebih besar. Metrik ini penting untuk memahami apakah model memiliki prediksi yang jauh dari nilai aktual pada data yang lebih besar.
   
3. **R-squared (R²) Score**:
   - R² mengukur proporsi variansi dalam data yang dapat dijelaskan oleh model. Nilai R² yang lebih tinggi (mendekati 1) menunjukkan bahwa model dapat menjelaskan sebagian besar variansi dalam data.

#### Hasil Evaluasi Model

- **MAE (0.0235)**: MAE yang relatif kecil menunjukkan bahwa prediksi model cukup mendekati nilai aktual pada rata-rata.
- **RMSE (0.0347)**: RMSE yang rendah menunjukkan bahwa model LSTM memiliki kesalahan prediksi yang minim, dan dapat memproyeksikan harga saham dengan ketepatan yang baik dalam satuan harga saham.
- **R² Score (0.9008)**: Nilai R² yang sangat tinggi (90%) menunjukkan bahwa model LSTM berhasil menjelaskan hampir seluruh variasi data harga saham BBRI, menandakan kemampuan model dalam menangkap pola harga saham jangka panjang.

#### Kesimpulan

Model LSTM yang dibangun berhasil memenuhi tujuan prediksi harga saham BBRI. Dengan **MAE** dan **RMSE** yang rendah serta **R²** yang sangat tinggi, model ini menunjukkan performa yang sangat baik dalam memprediksi harga penutupan saham berdasarkan data historis. Model ini sangat efektif dalam menangkap pola jangka panjang pergerakan harga saham BBRI dan memberikan hasil yang akurat untuk penggunaan lebih lanjut dalam proyeksi harga saham di masa depan.

## Reference
[[1]](https://journal.walisongo.ac.id/index.php/square/article/view/5626)
[[2]](https://medium.com/@prajjwalchauhan94017/stock-prediction-and-forecasting-using-lstm-long-short-term-memory-9ff56625de73)
[[3]](https://www.publish.ojs-indonesia.com/index.php/SIBATIK/article/view/798)
