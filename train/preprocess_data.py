import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

# CSV dosyasını oku
df = pd.read_csv("train/malicious_url_dataset.csv")

# Sadece 'url' ve 'type' sütunlarını al
df = df[['url', 'type']]

# Etiketleri sayısallaştır (Label Encoding)
df['label'] = df['type'].astype('category').cat.codes
# Örneğin: benign = 0, defacement = 1, phishing = 2

# URL ve etiketleri ayır
X = df['url']
y = df['label']

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Eğitim ve test verisine böl
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Verileri ve TF-IDF modelini kaydet
os.makedirs("train", exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), "train/processed_data.pkl")
joblib.dump(vectorizer, "train/tfidf_vectorizer.pkl")

print("✅ Veri başarıyla işlendi ve kaydedildi.")
