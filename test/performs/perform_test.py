import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    """Test that the model achieves at least 80% accuracy on the test set"""
    # Load the dataset
    data_path = os.path.join(os.path.dirname(__file__), "../../data", "spam.csv")
    df = pd.read_csv(data_path)
    X, y = df["Message"], df["Category"]
    
    # Load the trained model and vectorizer
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    model = joblib.load(os.path.join(models_dir, "spam_model.pkl"))
    vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    
    # Vectorize the text data
    X_vectorized = vectorizer.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_vectorized)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    
    # Assert minimum accuracy threshold
    assert accuracy >= 0.8, f"Model accuracy too low: {accuracy:.4f}. Expected >= 0.8"
    
    print(f"Model accuracy: {accuracy:.4f}")

def test_model_performance_on_spam():
    """Test that the model correctly identifies spam messages"""
    # Load the dataset
    data_path = os.path.join(os.path.dirname(__file__), "../../data", "spam.csv")
    df = pd.read_csv(data_path)
    
    # Filter only spam messages
    spam_messages = df[df["Category"] == "spam"]["Message"]
    
    # Load the trained model and vectorizer
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    model = joblib.load(os.path.join(models_dir, "spam_model.pkl"))
    vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    
    # Vectorize and predict
    X_vectorized = vectorizer.transform(spam_messages)
    predictions = model.predict(X_vectorized)
    
    # Calculate spam detection rate
    spam_detection_rate = (predictions == "spam").mean()
    
    # Assert that at least 90% of spam messages are correctly identified
    assert spam_detection_rate >= 0.9, f"Spam detection rate too low: {spam_detection_rate:.4f}. Expected >= 0.9"
    
    print(f"Spam detection rate: {spam_detection_rate:.4f}")

def test_model_performance_on_ham():
    """Test that the model correctly identifies ham (non-spam) messages"""
    # Load the dataset
    data_path = os.path.join(os.path.dirname(__file__), "../../data", "spam.csv")
    df = pd.read_csv(data_path)
    
    # Filter only ham messages
    ham_messages = df[df["Category"] == "ham"]["Message"]
    
    # Load the trained model and vectorizer
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    model = joblib.load(os.path.join(models_dir, "spam_model.pkl"))
    vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    
    # Vectorize and predict
    X_vectorized = vectorizer.transform(ham_messages)
    predictions = model.predict(X_vectorized)
    
    # Calculate ham detection rate
    ham_detection_rate = (predictions == "ham").mean()
    
    # Assert that at least 95% of ham messages are correctly identified
    assert ham_detection_rate >= 0.95, f"Ham detection rate too low: {ham_detection_rate:.4f}. Expected >= 0.95"
    
    print(f"Ham detection rate: {ham_detection_rate:.4f}")
