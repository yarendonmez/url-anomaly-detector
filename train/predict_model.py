import joblib

# Model ve TF-IDF yükle
model = joblib.load("train/model.pkl")
vectorizer = joblib.load("train/tfidf_vectorizer.pkl")

# Sınıf etiketleri (şu anki sıraya göre)
class_labels = {
    0: "Benign (Güvenli)",
    1: "Phishing (Oltalama)",
    2: "Defacement (Tahrifat)",
    3: "Malware / Other"
}

def predict_url(url):
    # URL'yi vektöre çevir
    url_vect = vectorizer.transform([url])
    
    # Tahmin yap
    prediction = model.predict(url_vect)[0]
    
    # Sonucu yazdır
    print(f"🔗 URL: {url}")
    print(f"⚠️ Tahmin: {class_labels.get(prediction, 'Bilinmeyen')}")

# Test
if __name__ == "__main__":
    test_url = input("URL girin: ")
    predict_url(test_url)
