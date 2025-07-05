# Part 3: Ethics & Optimization

## 1. Ethical Considerations

### Identify potential biases in your MNIST or Amazon Reviews model. How could tools like TensorFlow Fairness Indicators or spaCy's rule-based systems mitigate these biases?

#### **Bias in the MNIST Model**

While the MNIST dataset is fairly clean and balanced, potential biases can still arise:

*   **Handwriting Styles:** The model might be biased towards certain handwriting styles that are more common in the training data. For example, if the training set contains digits written primarily by people from a specific demographic or region, it might perform worse on handwriting from underrepresented groups.
*   **Data Imbalance (if it existed):** If the dataset were imbalanced (e.g., fewer examples of the digit '5'), the model might become less accurate at classifying that digit.

**Mitigation with TensorFlow Fairness Indicators:**

*   **Evaluation:** TensorFlow Fairness Indicators can be used to evaluate the model's performance across different subgroups of the data. To do this, we would need to slice the data based on some features (e.g., if we had metadata about the writers).
*   **Identification:** By visualizing fairness metrics like accuracy, precision, and recall for each subgroup, we could identify performance disparities. For instance, we could see if the model is less accurate for a specific handwriting style.
*   **Remediation:** Once a bias is identified, we could use techniques like data augmentation (e.g., rotating, shifting digits) or re-sampling to balance the dataset and improve the model's fairness.

#### **Bias in the Amazon Reviews Model (spaCy)**

Our simple rule-based sentiment model for Amazon reviews is highly susceptible to bias:

*   **Keyword-based Bias:** The sentiment is determined by a small, manually curated list of positive and negative words. This is very simplistic and can easily be wrong. For example, the word "hot" was labeled as negative (implying overheating), but in another context ("this new phone is hot!"), it could be positive.
*   **Sarcasm and Nuance:** The model cannot understand sarcasm or complex linguistic nuances. A review like "Brilliant, my phone broke in just two days" would be incorrectly classified as positive.
*   **Demographic Bias:** The pre-trained NER models from spaCy, while powerful, are trained on large text corpora (like news articles, Wikipedia). They might be less accurate at identifying product names or brands that are specific to certain cultures or demographics if those are not well-represented in the training data.

**Mitigation with spaCy's Rule-Based Systems:**

*   **Improving Sentiment Rules:** We can create more sophisticated rule-based systems using spaCy's `Matcher` or `PhraseMatcher`. Instead of single keywords, we can define patterns. For example, a pattern could look for a negative word followed by a product name, or a positive word in proximity to a verb like "love" or "like".
*   **Customizing NER:** If we find that the pre-trained NER model is failing on specific product names, we can use spaCy to train a custom NER model. We would need to create a labeled dataset with examples of the entities we want to recognize. This allows the model to learn the specific context of our domain (e.g., Amazon reviews).
*   **Combining Rules and ML:** For a more robust solution, we can use a hybrid approach. The rule-based system can handle clear-cut cases, while a machine learning model (trained on labeled review data) can handle the more ambiguous ones. This can help to reduce the biases inherent in a purely rule-based or purely ML-based system. 