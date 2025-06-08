import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Sample tweets (replace with real data via Twitter API)
tweets = [
    "I love the new iPhone! It's amazing 😍",
    "This is the worst update ever. #fail",
    "Meh, the new features are okay, nothing special.",
    "Absolutely fantastic customer service!",
    "I'm not happy with the battery life.",
]

# Preprocess and analyze
def clean_tweet(tweet):
    tweet = re.sub(r"http\\S+|@\\S+|#\\S+", "", tweet)
    tweet = re.sub(r"[^A-Za-z\\s]", "", tweet)
    return tweet.lower().strip()

results = []
for tweet in tweets:
    cleaned = clean_tweet(tweet)
    score = sid.polarity_scores(cleaned)
    compound = score['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    results.append((tweet, sentiment))

# Output
df = pd.DataFrame(results, columns=["Tweet", "Sentiment"])
print(df)

# Visualize
df['Sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution', color=['green', 'red', 'gray'])
plt.show()
