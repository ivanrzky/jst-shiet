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
- Label 9 = Huruf J

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
- **Epochs**: 10 (default, dapat diubah di Web UI)
- **Batch Size**: 128 (default, dapat diubah di Web UI)
- **Validation Split**: 10% dari data training

## Struktur File

```
project/
├── A_Z Handwritten Data.csv   # Dataset
├── main.py                    # Script original (standalone)
├── model.py                   # Modular functions untuk CNN
├── app.py                     # Streamlit Web UI
├── requirements.txt           # Dependencies
└── README.md                  # Dokumentasi
```

### Deskripsi File:

| File | Deskripsi |
|------|-----------|
| `model.py` | Modular functions: data loading, model building, training, prediction |
| `app.py` | Streamlit Web UI dengan full features |
| `main.py` | Script original untuk testing standalone |
| `requirements.txt` | Dependencies dengan version pinning |

## Cara Menggunakan

### Option 1: Menggunakan Web UI (Streamlit) - RECOMMENDED

1. Pastikan dataset `A_Z Handwritten Data.csv` berada di folder yang sama

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan Streamlit app:
```bash
streamlit run app.py
```

4. Buka browser di `http://localhost:8501`

### Option 2: Menggunakan Script Original

1. Pastikan dataset `A_Z Handwritten Data.csv` berada di folder yang sama

2. Install dependencies:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

3. Jalankan script:
```bash
python main.py
```

## Web UI Features

### Sidebar Controls
- **Epochs Slider**: Atur jumlah epoch untuk training (1-50)
- **Batch Size**: Pilih batch size (32, 64, 128, 256)
- **Train Model Button**: Mulai training dengan progress bar
- **Test Index Input**: Pilih index gambar dari test set (dengan validasi)
- **Random Index Button**: Generate index random untuk testing
- **Predict Button**: Jalankan prediksi
- **Model Status**: Tampilkan status model (trained/untrained) dan accuracy

### Main Display
- **Test Image Display**: Tampilkan gambar 28x28 grayscale
- **Prediction Results**: True label vs Predicted label dengan indicator (Correct/Wrong)
- **Confidence Score**: Persentase confidence dengan progress bar

### Visualization (Plotly Interactive Charts)
- **Confidence Bar Chart**: Chart interaktif untuk semua 10 kelas (A-J)
- **Training History**: Grafik Accuracy dan Loss per epoch (dual chart)
- **Final Metrics**: Test accuracy, validation accuracy, train accuracy, loss

### Optimizations
- **Data Caching**: `@st.cache_data` untuk loading data sekali saja
- **Session State**: Menyimpan model, history, dan prediction results
- **Progress Callback**: Real-time training progress dengan Keras callback
- **Environment Variables**: Suppress TensorFlow warnings

## Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0,<2.0.0
tensorflow>=2.10.0,<2.16.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.17.0
```

## Troubleshooting

### Error: ImportError keras/tensorflow

Jika mendapat error seperti:
```
ImportError: cannot import name 'MobileNetV3Large' from 'keras.applications.mobilenet_v3'
```

**Solusi:**
```bash
# Uninstall dan reinstall tensorflow
pip uninstall keras tensorflow -y
pip install tensorflow==2.15.0

# Atau reinstall semua dependencies
pip install -r requirements.txt --force-reinstall
```

### Error: TF_ENABLE_ONEDNN_OPTS Warning

Warning seperti:
```
oneDNN custom operations are on. You may see slightly different numerical results...
```

**Solusi:** Warning ini sudah di-suppress di code. Jika masih muncul, tambahkan environment variable:
```bash
# Windows PowerShell
$env:TF_ENABLE_ONEDNN_OPTS="0"

# Linux/Mac
export TF_ENABLE_ONEDNN_OPTS=0
```

### Error: Dataset tidak ditemukan

```
Dataset tidak ditemukan! Pastikan file 'A_Z Handwritten Data.csv' ada di folder yang sama.
```

**Solusi:** Pastikan file `A_Z Handwritten Data.csv` berada di folder yang sama dengan `app.py`.

## Output

### Web UI (app.py)
- Interactive dashboard dengan sidebar controls
- Real-time training progress dengan callback
- Visualisasi prediksi dengan confidence scores (Plotly)
- Grafik training history (accuracy & loss)
- Model status dan final metrics

### Script Original (main.py)
- Shape data setelah preprocessing
- Distribusi data per kelas
- Summary model
- Training progress
- Test accuracy
- Visualisasi accuracy dan loss per epoch (matplotlib)
- Contoh prediksi pada gambar test

## Screenshots

### Web UI Layout
```
+------------------------------------------+
|  SIDEBAR          |  MAIN CONTENT        |
|                   |                      |
| [Training Config] |  [Image Display]     |
|  - Epochs slider  |  28x28 grayscale     |
|  - Batch size     |                      |
|  - Train button   |  [Prediction Info]   |
|                   |  True: A             |
| [Prediction]      |  Pred: A (Correct)   |
|  - Index input    |  Confidence: 98.5%   |
|  - Random btn     |                      |
|  - Predict btn    |  [Confidence Chart]  |
|                   |  Bar chart A-J       |
| [Model Status]    |                      |
|  - Trained: Yes   |  [Training History]  |
|  - Accuracy: 95%  |  Acc & Loss graphs   |
+------------------------------------------+
```

## License

Project ini dibuat untuk keperluan pembelajaran mata kuliah Jaringan Saraf Tiruan (JST).
