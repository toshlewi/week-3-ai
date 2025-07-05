# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. Load and Preprocess the Data ---

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to a pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

# Check for missing values (Iris dataset is clean, but this is a good practice)
print("Checking for missing values:")
print(df.isnull().sum())
print("-" * 30)

# Encode labels (The Iris dataset is already encoded, but we'll include the code for demonstration)
# If the labels were strings, we would use LabelEncoder
# For example:
# string_labels = ['setosa', 'versicolor', 'virginica']
# encoder = LabelEncoder()
# encoded_labels = encoder.fit_transform(string_labels)
# In our case, `y` is already 0, 1, 2, which is what LabelEncoder would produce.
print(f"Target classes are already encoded: {target_names} -> {pd.Series(y).unique()}")
print("-" * 30)

# --- 2. Train the Decision Tree Classifier ---

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("-" * 30)

# Initialize the Decision Tree Classifier
# We use random_state for reproducibility
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
print("Training the Decision Tree model...")
dt_classifier.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)

# --- 3. Evaluate the Model ---

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
# For multi-class classification, we specify the 'average' parameter for precision and recall.
# 'weighted' accounts for label imbalance.
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print("-" * 30)

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names)) 