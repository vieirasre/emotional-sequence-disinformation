import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline
import nltk

import emoji
import demoji

import nltk
from nltk.corpus import stopwords
# Baixar as stop words do nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Certifique-se de baixar os dados do demoji
demoji.download_codes()

# FUNCTIONS ·

# EMOJI TREATMENT #
# emoji recognition and count
def emoji_count(tweet):
    tweet = emoji.demojize(tweet, delimiters=('__','__'))
    pattern = r'_+[a-z_&]+_+'
    return len(re.findall(pattern, tweet))

# emoji replacement
def replace_emojis_with_descriptions(text):
    # Substitui emojis pelas descrições padrão do demoji
    return demoji.replace_with_desc(text, sep=",")

# CLEANING #

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links and special characters
    (except punctuation, apostrophes, and monetary symbols) using simple regex statements.
    '''
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t\U0001F600-\U0001F64F.,!?':;’$€£])|(\w+:\/\/\S+)", " ", tweet).split())

# function to remove stopwords to apply word cloud
def clean_text(text):
    # Remover stop words
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text

# SENTIMENT ANALYSIS #
# TextBlob
def classify_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Vader
def classify_sentiment_vader(polarity_scores):
    '''
    Classify sentiment based on the compound score from vaderSentiment polarity scores.
    '''
    compound = polarity_scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'