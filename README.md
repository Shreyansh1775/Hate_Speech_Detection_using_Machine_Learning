# 🚫 Hate Speech Detection Web App

This is a simple web application that detects hate speech in text using a machine learning model trained on Twitter data. The app uses a logistic regression classifier with TF-IDF features, and is built with Flask for the backend and HTML/CSS for the frontend.

---

## 🧠 What It Does

You enter some text — the app analyzes it and tells you whether it contains hate speech or not. It's a binary classification model (Hate/Offensive vs. Not Hate).

This can be a useful demo for NLP text classification, hate speech moderation tools, or just experimenting with ML + web app integration.

---

## 🗂️ Project Structure

```
project/
├── static/
│   └── style.css              # Frontend styling
├── templates/
│   └── index.html             # Main webpage template
├── Hate_Speech_App.py         # Flask application
├── Hate_Speech_Detection.py   # Model training and evaluation
├── twitter.csv                # Dataset used for training
├── logreg_model.pkl           # Trained logistic regression model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── best_model.pkl             # (Optional) Alternative model
```

---

## 🛠️ How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/hate-speech-detector.git
   cd hate-speech-detector
   ```

2. **Install dependencies**

   It's recommended to use a virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, here’s the basic list:

   ```bash
   pip install flask nltk scikit-learn imbalanced-learn pandas matplotlib seaborn wordcloud
   ```

3. **Download required NLTK data**

   Run Python and execute:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

4. **Start the Flask app**

   ```bash
   python Hate_Speech_App.py
   ```

5. **Visit in browser**

   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to try it out.

---

## 🧪 Training the Model (Optional)

If you want to retrain or tweak the model, run:

```bash
python Hate_Speech_Detection.py
```

It’ll:

- Load the Twitter dataset
- Clean and preprocess the text
- Vectorize with TF-IDF
- Handle class imbalance with SMOTE
- Train a logistic regression model
- Save the model and vectorizer as `.pkl` files

---

## 🌐 Deploying

You can deploy this on:

- [Render](https://render.com)
- [Replit](https://replit.com)
- [Heroku](https://heroku.com)

Just make sure:

- Your `index.html` is in a `templates/` folder
- Your `style.css` is in a `static/` folder
- All required files are in the root or correctly linked
- You set `web: python Hate_Speech_App.py` as the start command in `Procfile` (if needed)

---

## 📌 Notes

- This is a basic binary classifier and should not be used for serious moderation without improvements.
- You can experiment by replacing the model with `best_model.pkl` if you trained a better one.

---

## 📬 Contact

If you find this useful or have suggestions, feel free to reach out or open an issue.
