import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Preprocess text
def preprocess(text):
    """Cleans text by converting to lowercase and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text


# Sentiment word lists
positive_words = {"love", "great", "fantastic", "amazing", "happy", "good", "awesome", "enjoy"}
negative_words = {"hate", "worst", "bad", "terrible", "disappointing", "awful", "horrible", "worried"}


# Define a simple scoring function
def analyze_sentiment(review):
    """
    Analyze sentiment by counting occurrences of positive and negative words.
    Returns 'Positive', 'Negative', or 'Neutral' based on word counts.
    """
    # Preprocess the review text
    review = preprocess(review)
    
    # Tokenize and split text into words
    words = review.split()
    
    # Count positive and negative words
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Assign sentiment based on word counts
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
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
        "Really good product."

        # Negative reviews
        "I hate this product. Terrible experience.",
        "Disappointing product, very bad.",
        "Not worth the price, bad performance."
        
        # Neutral reviews
        "The product arrived in a box with standard delivery.",
        "I used the product as expected and it works fine.",
        "The weather was okay during my trip.",
        "I went to the store to buy groceries today.",
        "The meeting went fine with no complaints from anyone."
    ]
})

# Analyze sentiments
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

# Heatmap (visualizing sentiment matrix for insights)
conf_matrix = pd.crosstab(data["sentiment"], data["sentiment"])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Sentiment Heatmap")
plt.show()
