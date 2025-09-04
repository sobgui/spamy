import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# We plan to train a model to detect spam messages.
# To do this, we need to load the dataset and split it into training and testing sets.
# Then, we can train the model and save it.

print("Loading dataset...")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data", "spam.csv"))
print(f"Dataset loaded: {len(df)} messages")

# Check class distribution
print("\nClass distribution:")
print(df["Category"].value_counts())
print(f"Spam percentage: {df['Category'].value_counts()['spam'] / len(df) * 100:.2f}%")

X = df["Message"]  # Text
y = df["Category"] # Label

# Split the data into training and testing sets
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# Check training set distribution
print("Training set distribution:")
print(y_train.value_counts())

# Vectorize the text using improved TF-IDF
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increased from 5000
    stop_words='english',
    ngram_range=(1, 2),  # Use both unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
    lowercase=True,
    strip_accents='unicode'
)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_vectorized.shape}")

# Train the model with improved parameters
print("\nTraining model...")
model = LogisticRegression(
    random_state=42, 
    max_iter=2000,  # Increased iterations
    C=0.1,  # Regularization parameter (smaller = more regularization)
    class_weight='balanced',  # Handle class imbalance
    solver='liblinear'  # Better for small datasets
)
model.fit(X_train_vectorized, y_train)

# Cross-validation to check model stability
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=5, scoring='f1_macro')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluate the model
print("\nEvaluating model...")
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate specific metrics
spam_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
spam_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
spam_f1 = 2 * (spam_precision * spam_recall) / (spam_precision + spam_recall) if (spam_precision + spam_recall) > 0 else 0

print(f"\nSpam Detection Metrics:")
print(f"Precision: {spam_precision:.4f}")
print(f"Recall: {spam_recall:.4f}")
print(f"F1-Score: {spam_f1:.4f}")

# Save the model and vectorizer
print("\nSaving model...")
models_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))

print("Model and vectorizer saved successfully!")

