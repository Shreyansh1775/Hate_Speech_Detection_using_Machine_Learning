import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import pickle
import warnings
import os

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # Handle class imbalance

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Ensure dataset exists
dataset_path = "twitter.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset '{dataset_path}' not found! Please provide a valid dataset.")

# Load dataset
tweet_df = pd.read_csv(dataset_path)

# Display columns to debug structure
print("Columns in dataset:", tweet_df.columns)

# Rename column 'class' to 'label' (if needed)
if 'class' in tweet_df.columns:
    tweet_df.rename(columns={'class': 'label'}, inplace=True)

# Ensure dataset has 'tweet' and 'label' columns
if 'tweet' not in tweet_df.columns or 'label' not in tweet_df.columns:
    raise KeyError("Dataset is missing 'tweet' or 'label' column! Check CSV formatting.")

# Check label distribution
print("Label distribution before mapping:\n", tweet_df['label'].value_counts())

# Convert multi-class labels into binary (if needed)
# Assuming 0 = Hate Speech, 1 = Offensive, 2 = Neither
tweet_df['label'] = tweet_df['label'].apply(lambda x: 1 if x in [0, 1] else 0)

# Check updated label distribution
print("Label distribution after mapping:\n", tweet_df['label'].value_counts())

# Drop missing values
tweet_df.dropna(inplace=True)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https?://\S+|www\.\S+", '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+|#', '', tweet)  # Remove mentions and hashtags
    tweet = re.sub(r'[^a-z\s]', '', tweet)  # Remove non-alphabetic characters
    tweet_tokens = nltk.word_tokenize(tweet)  # Tokenization
    filtered_tweet = [lemmatizer.lemmatize(w) for w in tweet_tokens if w not in stop_words]
    return " ".join(filtered_tweet)

# Apply preprocessing
tweet_df['clean_tweet'] = tweet_df['tweet'].apply(preprocess_text)

# Drop duplicates
tweet_df = tweet_df.drop_duplicates(subset='clean_tweet')

# Visualize class distribution
plt.figure(figsize=(5, 5))
sns.countplot(x='label', data=tweet_df)
plt.title("Class Distribution")
plt.show(block=False)

# Generate word clouds
def generate_wordcloud(data, title):
    text = ' '.join(data)
    plt.figure(figsize=(10, 5))
    wordcloud = WordCloud(max_words=500, width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14)
    plt.show(block=False)

# Word cloud for non-hate speech
generate_wordcloud(tweet_df[tweet_df.label == 0]['clean_tweet'], "Most frequent words in non-hate tweets")

# Word cloud for hate speech
generate_wordcloud(tweet_df[tweet_df.label == 1]['clean_tweet'], "Most frequent words in hate/offensive tweets")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_features=5000)
X = tfidf.fit_transform(tweet_df['clean_tweet'])
Y = tweet_df['label']

# Handle Class Imbalance Using SMOTE (only on training data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Logistic Regression model with class weights
model = LogisticRegression(class_weight='balanced', max_iter=200)
model.fit(X_train_resampled, y_train_resampled)

# Save Model and Vectorizer
pickle.dump(model, open('logreg_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hate", "Hate"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
