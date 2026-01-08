import pandas as pd
import numpy as np

# Ganti namanya kalau beda
data = pd.read_csv("A_Z Handwritten Data.csv").astype("float32")
print("data shape:", data.shape)

# cek label mentah di CSV (kolom pertama)
raw_labels = data.iloc[:, 0].to_numpy()
print("unique raw labels:", np.unique(raw_labels))

mask = raw_labels < 10
data_filtered = data[mask]

labels = data_filtered.iloc[:, 0].to_numpy().astype(int)
pixels = data_filtered.iloc[:, 1:].to_numpy()

print("Filtered shape:", data_filtered.shape)

x = pixels.reshape(-1, 28, 28, 1) / 255.0
print("x shape:", x.shape)

alpha = list("ABCDEFGHIJ")

unique, counts = np.unique(labels, return_counts=True)
print("unique labels:", unique)
print("counts:", counts)

for label, count in zip(unique, counts):
    huruf = alpha[int(label)]
    print(huruf, ":", count)

# Bentuk jadi (n_samples, 28, 28, 1) dan normalisasi ke 0–1
x = pixels.reshape(-1, 28, 28, 1) / 255.0  # tambahkan channel=1
print("x shape:", x.shape)

# One-hot encoding: 0..9 -> vektor 10 dimensi
from tensorflow.keras.utils import to_categorical
y = to_categorical(labels, num_classes=10)
print("y shape:", y.shape)


# ================= BALANCING DATA =================
from collections import Counter

min_count = min(Counter(labels).values())
print("Data per kelas disamakan menjadi:", min_count)

x_balanced = []
y_balanced = []
labels_balanced = []

for i in range(10):
    idx = np.where(labels == i)[0][:min_count]
    x_balanced.append(x[idx])
    y_balanced.append(y[idx])
    labels_balanced.append(labels[idx])

x = np.concatenate(x_balanced)
y = np.concatenate(y_balanced)
labels = np.concatenate(labels_balanced)

print("Balanced shape:", x.shape)

# Split data into train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, labels_train, labels_test = train_test_split(
    x, y, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

from tensorflow.keras import layers, models

#model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')   # 10 kelas: A–J
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    x_train, y_train,
    epochs=10,          # nanti bisa dinaikkan kalau masih kuat
    batch_size=128,
    validation_split=0.1,   # 10% dari train buat validasi
    verbose=1
)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)


import matplotlib.pyplot as plt

alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # buat mapping index -> huruf

idx =0 # boleh ganti angka lain
img = x_test[idx]

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.axis('off')

pred = model.predict(img.reshape(1, 28, 28, 1))
pred_label = np.argmax(pred)

print("Label asli     :", alpha[labels_test[idx]])
print("Prediksi model :", alpha[pred_label])


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = labels_test


#untuk menambah akurasi testing

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
   
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
