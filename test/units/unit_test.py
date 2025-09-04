import os

def test_model_exists():
    """Test if the trained spam model exists"""
    model_path = os.path.join(os.path.dirname(__file__), "../../models", "spam_model.pkl")
    assert os.path.exists(model_path), "Spam model file should exist"

def test_vectorizer_exists():
    """Test if the vectorizer exists"""
    vectorizer_path = os.path.join(os.path.dirname(__file__), "../../models", "vectorizer.pkl")
    assert os.path.exists(vectorizer_path), "Vectorizer file should exist"

def test_models_directory_exists():
    """Test if the models directory exists"""
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    assert os.path.exists(models_dir), "Models directory should exist"
