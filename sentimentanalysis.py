import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from textblob import TextBlob
import nltk

# Preprocess text
def preprocess(text):
    """Cleans text by converting to lowercase and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text


# Analyze sentiment with TextBlob
def analyze_sentiment(review):
    """
    Analyze sentiment using TextBlob.
    TextBlob returns a polarity value between -1 and 1.
    - Polarity < 0 indicates Negative sentiment.
    - Polarity = 0 indicates Neutral sentiment.
    - Polarity > 0 indicates Positive sentiment.
    """
    review = preprocess(review)  # Preprocess the review text
    blob = TextBlob(review)
    
    # Classify sentiment based on polarity
    if blob.sentiment.polarity > 0:
        return "Positive"
    elif blob.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


# Create a dataset with additional neutral examples
data = pd.DataFrame({
    "review": [
        # Positive reviews
        "I love this product, it works great!",
        "This is the best experience I have ever had.",
        "Amazing quality and very fast shipping.",
        "I had an amazing experience, very happy.",
        "Really good product.",

        # Negative reviews
        "I hate this product. Terrible experience.",
        "Disappointing product, very bad.",
        "Not worth the price, bad performance.",

        # Neutral reviews
        "The product arrived in a box with standard delivery.",
        "I used the product as expected and it works fine.",
        "The weather was okay during my trip.",
        "I went to the store to buy groceries today.",
        "The meeting went fine with no complaints from anyone."
    ]
})

# Analyze sentiment
data["sentiment"] = data["review"].apply(analyze_sentiment)

# Map sentiment categories to numerical values for visualization
label_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
data["sentiment_label"] = data["sentiment"].map(label_mapping)

# Display results
print(data)

# Plotting sentiment distribution
sns.countplot(x="sentiment", data=data, hue="sentiment", palette="viridis", legend=False)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
