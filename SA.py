import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK resources are downloaded
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

# Set plot style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# --- 1. Simulate Data Collection: Diverse Text Data ---
# This dataset represents a mix of reviews, social media comments, and news-like snippets.
data = {
    'TextID': range(1, 15),
    'Text': [
        "This product is absolutely amazing! Highly recommend it to everyone.", # Positive review
        "The service was terrible and I'm very disappointed with my experience.", # Negative review
        "It's an interesting article, but I don't have strong feelings either way.", # Neutral news
        "Just received my order, everything looks perfect! So happy!", # Positive social media
        "The new policy might have some unforeseen consequences for citizens.", # Neutral news
        "Feeling so sad today, nothing seems to go right.", # Negative emotion
        "This movie made me laugh out loud! Pure joy!", # Positive emotion
        "I am furious about the recent price hike! Unacceptable.", # Negative emotion
        "The weather today is just lovely, perfect for a walk.", # Positive general
        "The meeting concluded without any major decisions being made.", # Neutral general
        "What a frustrating bug! It keeps crashing my app.", # Negative technical
        "Thrilled with the new update, it fixed all my issues!", # Positive technical
        "This report provides a comprehensive overview of market trends.", # Neutral news/report
        "Absolutely disgusted by their unethical practices.", # Negative strong
        "Such a delightful surprise! Made my day." # Positive strong
    ]
}
df = pd.DataFrame(data)

print("--- Simulated Raw Text Data ---")
print(df)
print("\n" + "="*50 + "\n")

# --- 2. Data Preprocessing ---

# Function to clean text
def preprocess_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    tokens = word_tokenize(text) # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')] # Remove stopwords
    return ' '.join(tokens)

df['Cleaned_Text'] = df['Text'].apply(preprocess_text)

print("--- Text Data After Preprocessing ---")
print(df[['Text', 'Cleaned_Text']].head())
print("\n" + "="*50 + "\n")

# --- 3. Sentiment Analysis using TextBlob ---

# Function to get sentiment polarity and subjectivity
def get_sentiment_scores(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Apply sentiment analysis
df['Polarity'], df['Subjectivity'] = zip(*df['Cleaned_Text'].apply(get_sentiment_scores))

# Categorize sentiment based on polarity
def categorize_sentiment(polarity):
    if polarity > 0.1: # Threshold for positive
        return 'Positive'
    elif polarity < -0.1: # Threshold for negative
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Polarity'].apply(categorize_sentiment)

print("--- Text Data with Sentiment Analysis Results ---")
print(df[['Text', 'Polarity', 'Subjectivity', 'Sentiment']])
print("\n" + "="*50 + "\n")

# --- 4. Emotion Detection using Lexicons (Simplified Example) ---

# Define simple emotion lexicons
# In a real scenario, these would be much more extensive and curated.
emotion_lexicons = {
    'Joy': ['happy', 'joy', 'thrilled', 'delightful', 'amazing', 'perfect', 'love', 'great'],
    'Anger': ['furious', 'unacceptable', 'disgusted', 'hate', 'frustrating', 'terrible'],
    'Sadness': ['sad', 'disappointed', 'unhappy', 'nothing seems right'], # 'nothing seems right' as a phrase
    'Surprise': ['surprise', 'unexpected', 'wow'],
    'Fear': ['fear', 'scared', 'anxious', 'worried']
}

# Function to detect emotions
def detect_emotions(text, lexicons):
    detected_emotions = {emotion: 0 for emotion in lexicons.keys()}
    words = word_tokenize(text.lower()) # Tokenize for matching

    for emotion, keywords in lexicons.items():
        for keyword in keywords:
            if keyword in text.lower(): # Check for whole words or phrases
                detected_emotions[emotion] += 1
    return detected_emotions

# Apply emotion detection
df['Emotions'] = df['Text'].apply(lambda x: detect_emotions(x, emotion_lexicons))

# To display emotions more clearly, we can extract the dominant one or a list
def get_dominant_emotion(emotion_dict):
    if not emotion_dict:
        return 'None'
    # Find emotion with max count, if ties, take fi
