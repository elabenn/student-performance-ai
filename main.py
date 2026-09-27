from ucimlrepo import fetch_ucirepo
import pandas as pd

# UCI Student Performance veri setini al
student_performance = fetch_ucirepo(id=320)

# Veriyi DataFrame'e aktar
X = student_performance.data.features
y = student_performance.data.targets

print("Veri seti başarıyla yüklendi.")
print()
print("Özellikler:")
print(X.head())

print()
print("Hedef değişken:")
print(y.head())

print()
print("Veri boyutu:")
print(X.shape)
print()
print("Sütunlar:")
print(X.columns.tolist())

print()
print("Eksik değerler:")
print(X.isnull().sum())

print()
print("Veri tipi:")
print(X.dtypes)
print()
print("Hedef değişken istatistikleri:")
print(y.describe())

print()
print("G3 not dağılımı:")
print(y["G3"].value_counts().sort_index())
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(y["G3"], bins=20)
plt.xlabel("Final Notu (G3)")
plt.ylabel("Öğrenci Sayısı")
plt.title("Final Notlarının Dağılımı")
plt.grid(axis="y", alpha=0.3)
plt.show()
plt.figure(figsize=(8, 5))

study_groups = [
    y["G3"][X["studytime"] == i]
    for i in sorted(X["studytime"].unique())
]

plt.boxplot(
    study_groups,
    tick_labels=sorted(X["studytime"].unique())
)

plt.xlabel("Çalışma Süresi Kategorisi")
plt.ylabel("Final Notu (G3)")
plt.title("Çalışma Süresi ve Final Notu İlişkisi")
plt.grid(axis="y", alpha=0.3)

plt.show()
plt.figure(figsize=(8, 5))

plt.scatter(X["absences"], y["G3"], alpha=0.6)

plt.xlabel("Devamsızlık Sayısı")
plt.ylabel("Final Notu (G3)")
plt.title("Devamsızlık ve Final Notu İlişkisi")
plt.grid(alpha=0.3)

plt.show()
correlation = X["absences"].corr(y["G3"])

print()
print("Devamsızlık - Final Notu korelasyonu:")
print(correlation)
plt.figure(figsize=(8, 5))

failure_groups = [
    y["G3"][X["failures"] == i]
    for i in sorted(X["failures"].unique())
]

plt.boxplot(
    failure_groups,
    tick_labels=sorted(X["failures"].unique())
)

plt.xlabel("Önceki Başarısızlık Sayısı")
plt.ylabel("Final Notu (G3)")
plt.title("Önceki Başarısızlıklar ve Final Notu İlişkisi")
plt.grid(axis="y", alpha=0.3)

plt.show()
# Makine öğrenmesi için veri hazırlama
X_model = X.copy()
y_model = y["G3"]

print()
print("Modelde kullanılacak özellik sayısı:")
print(X_model.shape[1])

print()
print("Model hedefi:")
print("G3 - Final Notu")
X_model = pd.get_dummies(X_model, drop_first=True)

print()
print("Kodlama sonrası veri boyutu:")
print(X_model.shape)

print()
print("Veri tipleri:")
print(X_model.dtypes)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.2,
    random_state=42
)

print()
print("Eğitim verisi boyutu:")
print(X_train.shape)

print()
print("Test verisi boyutu:")
print(X_test.shape)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print()
print("Model başarıyla eğitildi.")
print("İlk 10 tahmin:")
print(y_pred[:10])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print()
print("Model değerlendirme sonuçları:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
# Özellik önemlerini inceleme
feature_importance = pd.Series(
    model.feature_importances_,
    index=X_model.columns
)

feature_importance = feature_importance.sort_values(ascending=False)

print()
print("En önemli özellikler:")
print(feature_importance.head(10))
plt.figure(figsize=(9, 6))

feature_importance.head(10).sort_values().plot(kind="barh")

plt.xlabel("Önem Skoru")
plt.ylabel("Özellik")
plt.title("Final Notu Tahmininde En Önemli 10 Özellik")
plt.grid(axis="x", alpha=0.3)

plt.show()
import tensorflow as tf

print()
print("TensorFlow sürümü:")
print(tf.__version__)
# TensorFlow ile yapay sinir ağı modeli

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Verileri TensorFlow için sayısal forma çevir
X_train_tf = X_train.astype("float32")
X_test_tf = X_test.astype("float32")

y_train_tf = y_train.astype("float32")
y_test_tf = y_test.astype("float32")

# Modeli oluştur
tf_model = keras.Sequential([
    layers.Input(shape=(X_train_tf.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

# Modeli hazırla
tf_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Modeli eğit
history = tf_model.fit(
    X_train_tf,
    y_train_tf,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

print()
print("TensorFlow modeli başarıyla eğitildi.")
# TensorFlow modelinin tahminleri
tf_pred = tf_model.predict(X_test_tf, verbose=0).flatten()

# TensorFlow modelini değerlendirme
tf_mae = mean_absolute_error(y_test_tf, tf_pred)
tf_rmse = np.sqrt(mean_squared_error(y_test_tf, tf_pred))
tf_r2 = r2_score(y_test_tf, tf_pred)

print()
print("TensorFlow model değerlendirme sonuçları:")
print("MAE:", tf_mae)
print("RMSE:", tf_rmse)
print("R²:", tf_r2)# TensorFlow modelini ölçeklenmiş verilerle geliştirme

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.astype("float32")
X_test_scaled = X_test_scaled.astype("float32")

# Geliştirilmiş sinir ağı
improved_tf_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

improved_tf_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Erken durdurma
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# Modeli eğit
improved_history = improved_tf_model.fit(
    X_train_scaled,
    y_train_tf,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

print()
print("Geliştirilmiş TensorFlow modeli başarıyla eğitildi.")

# Tahmin
improved_tf_pred = improved_tf_model.predict(
    X_test_scaled,
    verbose=0
).flatten()

# Değerlendirme
improved_tf_mae = mean_absolute_error(y_test_tf, improved_tf_pred)
improved_tf_rmse = np.sqrt(
    mean_squared_error(y_test_tf, improved_tf_pred)
)
improved_tf_r2 = r2_score(y_test_tf, improved_tf_pred)

print()
print("Geliştirilmiş TensorFlow model değerlendirme sonuçları:")
print("MAE:", improved_tf_mae)
print("RMSE:", improved_tf_rmse)
print("R²:", improved_tf_r2)
# Model karşılaştırması

model_names = [
    "Random Forest",
    "TensorFlow",
    "Geliştirilmiş TensorFlow"
]

r2_scores = [
    r2,
    tf_r2,
    improved_tf_r2
]

plt.figure(figsize=(9, 5))

plt.bar(model_names, r2_scores)

plt.xlabel("Model")
plt.ylabel("R² Skoru")
plt.title("Model Performans Karşılaştırması")
plt.grid(axis="y", alpha=0.3)

plt.show()
# Gerçek ve tahmin edilen notların karşılaştırılması

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Gerçek Final Notu (G3)")
plt.ylabel("Tahmin Edilen Final Notu")
plt.title("Gerçek ve Tahmin Edilen Final Notlarının Karşılaştırılması")

# Mükemmel tahmin çizgisi
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.grid(alpha=0.3)

plt.show()
# Tahmin hatalarının analizi

errors = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.hist(errors, bins=15)

plt.xlabel("Tahmin Hatası (Gerçek - Tahmin)")
plt.ylabel("Öğrenci Sayısı")
plt.title("Model Tahmin Hatalarının Dağılımı")
plt.grid(axis="y", alpha=0.3)

plt.show()
# Eğitilmiş Random Forest modelini kaydetme

import joblib

joblib.dump(model, "student_performance_model.pkl")

print()
print("Random Forest modeli başarıyla kaydedildi.")