import os
import joblib

def predict_spam(message):
    """
    Predict if a message is spam or ham
    
    Args:
        message (str): The text message to classify
        
    Returns:
        str: 'spam' or 'ham'
    """
    # Load the vectorizer and model
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    model = joblib.load(os.path.join(models_dir, "spam_model.pkl"))
    
    # Convert message to numbers using the vectorizer
    message_vectorized = vectorizer.transform([message])
    
    # Make prediction using the model
    prediction = model.predict(message_vectorized)[0]
    
    return prediction

# Example usage
if __name__ == "__main__":
    # Test messages
    test_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts!",
        "Hey, how are you doing today?",
        "WINNER!! You have been selected to receive a £900 prize reward!"
    ]
    
    print("Spam Detection Results:")
    print("-" * 50)
    
    for message in test_messages:
        prediction = predict_spam(message)
        print(f"Message: {message[:50]}...")
        print(f"Prediction: {prediction}")
        print("-" * 50)
