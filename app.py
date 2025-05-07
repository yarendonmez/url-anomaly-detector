from flask import Flask, render_template_string, request
import joblib

# Flask uygulaması başlat
app = Flask(__name__)

# Model ve vectorizer yükle
model = joblib.load("train/model.pkl")
vectorizer = joblib.load("train/tfidf_vectorizer.pkl")

# Etiketler
class_labels = {
    0: "✅ Güvenli (Benign)",
    1: "❌ Phishing (Oltalama)",
    2: "❌ Defacement (Tahrifat)",
    3: "⚠️ Malware / Other"
}

# HTML template
HTML = """
<!doctype html>
<title>URL Güvenlik Analizi</title>
<h2>🔍 URL Güvenli mi?</h2>
<form method="post">
  <input type="text" name="url" style="width: 300px;" placeholder="URL giriniz..." required>
  <input type="submit" value="Analiz Et">
</form>
{% if url %}
  <h3>🔗 URL: <code>{{ url }}</code></h3>
  <h3>Sonuç: {{ result }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    url = None
    if request.method == "POST":
        url = request.form["url"]
        vect = vectorizer.transform([url])
        prediction = model.predict(vect)[0]
        result = class_labels.get(prediction, "Bilinmeyen")
    return render_template_string(HTML, result=result, url=url)

if __name__ == "__main__":
    app.run(debug=True)
