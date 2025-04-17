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
* Mengevaluasi performa model dengan metrik akurasi regresi (MAE, RMSE dan F2-Score).

### Solution
Solusi dilakukan dengan pendekatan sebagai berikut:
* Mengambil data historis harga saham BBRI dari tahun 2016 hingga 2024.
* Melakukan preprocessing data, termasuk normalisasi menggunakan MinMaxScaler dan pembentukan sliding window sepanjang 60 hari untuk membentuk sekuens input LSTM.
* Membangun model LSTM dengan arsitektur sederhana dan melakukan pelatihan menggunakan data yang telah dibagi menjadi data training dan testing.
* Mengevaluasi model menggunakan metrik Mean Absolute Error (MAE), Mean Squared Error (MSE) dan F2-Score.
* Membandingkan nilai prediksi dan harga aktual pada data testing.

## Data Understanding
Sumber Data: Dataset diperoleh dari [[IDX]](https://github.com/wildangunawan/Dataset-Saham-IDX/blob/master/Saham/LQ45/BBRI.csv) dengan kode saham BBRI, yaitu saham dari perusahaan Bank Rakyat Indonesia, yang bergerak di sektor perbankan. Data diambil untuk periode 2019-07-29 hingga 2025-02-21, mencakup lebih dari 1.356 baris data harian perdagangan saham.

Struktur Dataset: Dataset ini memiliki beberapa kolom utama yang merepresentasikan informasi harga dan volume perdagangan saham pada setiap harinya. Berikut penjelasan tiap kolom:
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
  * Dalam proses ini, kita akan menangani nilai hilang yang terdapat dalam dataset. Penanganan nilai hilang dilakukan dengan mengisi nilai-nilai yang hilang menggunakan median dari kolom yang bersangkutan. Hal ini bertujuan      untuk menjaga integritas data dan memastikan bahwa model yang dibangun tidak terpengaruh oleh nilai yang hilang.
2. Pemilihan Fitur
  * Dari dataset yang tersedia, dipilih empat kolom numerik yang relevan, yaitu: open_price, high, low, dan close
3. Normalisasi Data
  * Data numerik dinormalisasi menggunakan teknik Min-Max Scaling, yang mengubah skala nilai fitur menjadi rentang [0, 1]. Proses ini dilakukan menggunakan MinMaxScaler dari pustaka scikit-learn.
4. Pembagian Data Latih dan Uji
  * Setelah pemilihan Fitur dan Normalisasi Data, dataset di split dengan ratio 80:20, yaitu 80% untuk training dan 20% untuk testing.
