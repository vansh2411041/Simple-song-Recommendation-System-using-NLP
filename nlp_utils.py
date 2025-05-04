import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize models
nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
hf_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class SongAnalyzer:
    @staticmethod
    def clean_text(text):
        """Enhanced text cleaning"""
        if pd.isna(text):
            return ""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        return text.strip()

    @staticmethod
    def analyze_with_textblob(text):
        """Traditional sentiment analysis"""
        analysis = TextBlob(text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity
        }

    @staticmethod
    def analyze_with_vader(text):
        """Rule-based sentiment analysis"""
        scores = sia.polarity_scores(text)
        return {
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": scores["compound"]
        }

    @staticmethod
    def analyze_with_transformers(text):
        """Deep learning sentiment analysis"""
        try:
            result = hf_sentiment(text[:512])[0]  # Truncate to model max length
            return {
                "label": result["label"],
                "score": result["score"]
            }
        except:
            return {"label": "ERROR", "score": 0}

    @staticmethod
    def analyze_with_spacy(text):
        """Linguistic feature extraction"""
        doc = nlp(text)
        positive_words = ["love", "happy", "joy", "amazing"]
        negative_words = ["hate", "sad", "pain", "terrible"]
        
        pos_count = sum(1 for token in doc if token.text in positive_words)
        neg_count = sum(1 for token in doc if token.text in negative_words)
        
        return {
            "positive_words": pos_count,
            "negative_words": neg_count,
            "sentiment_ratio": pos_count / (neg_count + 0.001)  
        }

    @staticmethod
    def comprehensive_analysis(lyrics):
        """Run all analyses and return combined results"""
        cleaned = SongAnalyzer.clean_text(lyrics)
        if not cleaned:
            return {"error": "Empty lyrics"}
        
        return {
            "textblob": SongAnalyzer.analyze_with_textblob(cleaned),
            "vader": SongAnalyzer.analyze_with_vader(cleaned),
            "transformers": SongAnalyzer.analyze_with_transformers(cleaned),
            "spacy": SongAnalyzer.analyze_with_spacy(cleaned)
        }

    @staticmethod
    def find_similar_songs(lyrics_list):
        """Enhanced similarity with multiple features"""
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=10000
        )
        cleaned_lyrics = lyrics_list.fillna("").apply(SongAnalyzer.clean_text)
        tfidf_matrix = tfidf.fit_transform(cleaned_lyrics)
        return cosine_similarity(tfidf_matrix)  