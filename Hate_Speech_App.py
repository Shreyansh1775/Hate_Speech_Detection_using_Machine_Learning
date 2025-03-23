import pickle
import re
import nltk
import numpy as np
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Load trained model and vectorizer
model = pickle.load(open("logreg_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters and numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Hate Speech Detection Function
def detect_hate_speech(text):
    cleaned_text = preprocess_text(text)  # Preprocess user input
    text_vectorized = vectorizer.transform([cleaned_text])  # Convert to vector
    prediction = model.predict(text_vectorized)[0]  # Predict using model
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

# API Route for Prediction
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({"prediction": "Error: No text entered!"})

    prediction = detect_hate_speech(user_text)
    return jsonify({"prediction": prediction})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
