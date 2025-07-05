# Import necessary libraries
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# --- Pre-computation ---
# Before running, you need to download the spaCy model:
# python -m spacy download en_core_web_sm

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading the spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# --- 1. Text Data: Sample Amazon Reviews ---
amazon_reviews = [
    "The new iPhone 14 Pro from Apple is amazing, but the battery life could be better.",
    "I bought a Samsung Galaxy S22 and it has a fantastic camera. Highly recommended!",
    "My Sony WH-1000XM5 headphones are the best for noise cancellation. A bit pricey, though.",
    "The Dell XPS 15 laptop has a stunning display, but it gets hot under load.",
    "I'm not happy with my purchase of the Google Pixel 7. The software is buggy.",
    "The Canon EOS R6 is a brilliant camera for both photos and videos. I love it!",
    "Amazon's own Echo Dot is a great smart speaker for the price. Alexa is very responsive."
]

# --- 2. Goal: NER and Sentiment Analysis ---

def analyze_reviews(reviews):
    """
    Performs NER and rule-based sentiment analysis on a list of reviews.
    """
    # Define simple keywords for rule-based sentiment
    positive_words = ['amazing', 'fantastic', 'best', 'stunning', 'brilliant', 'love', 'great', 'highly recommended', 'happy']
    negative_words = ['could be better', 'pricey', 'hot', 'not happy', 'buggy']

    print("--- Amazon Review Analysis ---")
    for i, review in enumerate(reviews):
        print(f"\n--- Review #{i+1} ---")
        print(f"Text: {review}")

        doc = nlp(review)

        # --- Named Entity Recognition (NER) ---
        print("\nNamed Entities:")
        entities_found = False
        for ent in doc.ents:
            # We are interested in Organizations (brands) and Products
            if ent.label_ in ["ORG", "PRODUCT"]:
                print(f"- Entity: {ent.text}, Label: {ent.label_}")
                entities_found = True
        
        if not entities_found:
            print("- No specific product names or brands found.")

        # --- Rule-based Sentiment Analysis ---
        sentiment = "Neutral"
        # Simple check: more positive or negative keywords?
        pos_count = sum([1 for word in positive_words if word in review.lower()])
        neg_count = sum([1 for word in negative_words if word in review.lower()])

        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        
        print(f"\nSentiment: {sentiment}")
        print("-" * 20)

# Run the analysis
analyze_reviews(amazon_reviews) 