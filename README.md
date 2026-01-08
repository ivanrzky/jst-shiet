# Handwritten Alphabet Recognition

Proyek ini menggunakan dataset A-Z Handwritten Data untuk melatih model CNN (Convolutional Neural Network) dalam mengenali huruf tulisan tangan.

## Dataset

### A-Z Handwritten Data.csv

Dataset ini berisi gambar tulisan tangan untuk seluruh alfabet A-Z.

#### Karakteristik Dataset:
- **Format**: CSV (Comma Separated Values)
- **Struktur**:
  - Kolom pertama: Label (0-25 untuk A-Z)
  - Kolom 2-785: Pixel values (28 × 28 = 784 pixel)
- **Ukuran Gambar**: 28 × 28 piksel
- **Nilai Pixel**: 0-255 (grayscale)

#### Mapping Label:
- Label 0 = Huruf A
- Label 1 = Huruf B
- Label 2 = Huruf C
- ...
- Label 25 = Huruf Z

#### Preprocessing yang Diterapkan:
1. **Filtering**: Dataset difilter untuk hanya menggunakan huruf A-J (label 0-9) untuk pembelajaran
2. **Normalisasi**: Nilai pixel dinormalisasi dari 0-255 menjadi 0-1 dengan membagi 255
3. **Reshaping**: Data diubah dari bentuk (n_samples, 784) menjadi (n_samples, 28, 28, 1) untuk input CNN
4. **Balancing**: Data di-balance agar setiap kelas memiliki jumlah sampel yang sama (menggunakan jumlah minimum dari setiap kelas)
5. **One-Hot Encoding**: Label dikonversi menjadi format one-hot encoding untuk klasifikasi multi-kelas

#### Train-Test Split:
- **Train**: 80% dari data
- **Test**: 20% dari data
- **Stratified**: Split dilakukan dengan stratifikasi untuk menjaga proporsi setiap kelas

## Model

Model CNN yang digunakan memiliki arsitektur:
- **Input**: (28, 28, 1) - gambar grayscale 28×28
- **Conv2D Layer 1**: 32 filter (3×3), activation ReLU
- **MaxPooling2D**: (2, 2)
- **Conv2D Layer 2**: 64 filter (3×3), activation ReLU
- **MaxPooling2D**: (2, 2)
- **Flatten Layer**
- **Dense Layer**: 128 neuron, activation ReLU
- **Output Layer**: 10 neuron (untuk 10 kelas A-J), activation softmax

### Hyperparameters:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 128
- **Validation Split**: 10% dari data training

## Cara Menggunakan

1. Pastikan dataset `A_Z Handwritten Data.csv` berada di folder yang sama dengan `main.py`
2. Install dependencies:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```
3. Jalankan script:
```bash
python main.py
```

## Output

Program akan menampilkan:
- Shape data setelah preprocessing
- Distribusi data per kelas
- Summary model
- Training progress
- Test accuracy
- Visualisasi accuracy dan loss per epoch
- Contoh prediksi pada gambar test

## Dependencies

- pandas
- numpy
- tensorflow
- scikit-learn
- matplotlib
