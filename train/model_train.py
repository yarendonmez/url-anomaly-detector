print("dosya çalıştı")
import joblib
print("joblib import edildi")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# İşlenmiş veriyi yükle
X_train, X_test, y_train, y_test = joblib.load("train/processed_data.pkl")
print("veri yüklendi")

# Modeli oluştur
print("Model oluşturuluyor...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# Eğitimi başlat
print("Eğitim başlıyor...")
model.fit(X_train, y_train)

# Test verisiyle doğruluk ölç
print("Tahmin yapılıyor...")
y_pred = model.predict(X_test)
print("🎯 Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\n📊 Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Modeli kaydet
joblib.dump(model, "train/model.pkl")
print("✅ Model başarıyla eğitildi ve kaydedildi.")
