import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .spam-prediction {
        background-color: #fff5f5;
        border: 2px solid #f44336;
        color: #d32f2f;
    }
    .spam-prediction h3 {
        color: #d32f2f !important;
        margin: 0 0 0.5rem 0;
    }
    .spam-prediction p {
        color: #d32f2f !important;
        margin: 0;
        font-weight: 500;
    }
    .ham-prediction {
        background-color: #f1f8e9;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .ham-prediction h3 {
        color: #2e7d32 !important;
        margin: 0 0 0.5rem 0;
    }
    .ham-prediction p {
        color: #2e7d32 !important;
        margin: 0;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load the trained model, vectorizer, and calculate metrics"""
    try:
        # Get current directory and navigate to project root
        current_dir = os.getcwd()
        if current_dir.endswith('src'):
            project_root = os.path.dirname(current_dir)
        else:
            project_root = current_dir
        
        # Load model and vectorizer
        models_dir = os.path.join(project_root, "models")
        model = joblib.load(os.path.join(models_dir, "spam_model.pkl"))
        vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
        
        # Load dataset for metrics calculation
        data_path = os.path.join(project_root, "data", "spam.csv")
        df = pd.read_csv(data_path)
        
        return model, vectorizer, df
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def calculate_metrics(model, vectorizer, df):
    """Calculate model performance metrics"""
    from sklearn.model_selection import train_test_split
    
    X = df["Message"]
    y = df["Category"]
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    
    # Vectorize and predict
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_size': len(X_test),
        'spam_count': sum(y_test == 'spam'),
        'ham_count': sum(y_test == 'ham')
    }

def predict_spam(message, model, vectorizer):
    """Predict if a message is spam"""
    if not message.strip():
        return None, None
    
    # Vectorize the message
    message_vectorized = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_vectorized)[0]
    probability = model.predict_proba(message_vectorized)[0]
    
    return prediction, probability

# Load model and data
model, vectorizer, df = load_model_and_data()

if model is not None and vectorizer is not None and df is not None:
    # Calculate metrics
    metrics = calculate_metrics(model, vectorizer, df)
    
    # Main header
    st.markdown('<h1 class="main-header">📧 Spam Detection App</h1>', unsafe_allow_html=True)
    
    # Sidebar with model information
    with st.sidebar:
        st.header("🤖 Model Information")
        
        # Model metrics
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
        
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        # Dataset information
        st.subheader("Dataset Info")
        st.write(f"**Total Messages:** {len(df):,}")
        st.write(f"**Spam Messages:** {sum(df['Category'] == 'spam'):,}")
        st.write(f"**Ham Messages:** {sum(df['Category'] == 'ham'):,}")
        st.write(f"**Test Set Size:** {metrics['test_size']:,}")
        
        # Model details
        st.subheader("Model Details")
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Vectorizer:** TF-IDF")
        st.write("**Features:** 10,000")
        st.write("**N-grams:** (1,2)")
        
        # Performance rating
        st.subheader("Performance Rating")
        if metrics['accuracy'] >= 0.95:
            st.success("🟢 Excellent")
        elif metrics['accuracy'] >= 0.90:
            st.info("🟡 Good")
        elif metrics['accuracy'] >= 0.80:
            st.warning("🟠 Fair")
        else:
            st.error("🔴 Needs Improvement")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Spam Detection")
        
        # Text input
        message = st.text_area(
            "Enter a message to check if it's spam:",
            placeholder="Type your message here...",
            height=150,
            help="Enter any text message and our AI model will determine if it's spam or not."
        )
        
        # Predict button
        if st.button("🔍 Check Message", type="primary"):
            if message.strip():
                prediction, probability = predict_spam(message, model, vectorizer)
                
                if prediction is not None:
                    spam_prob = probability[1]  # Probability of being spam
                    ham_prob = probability[0]   # Probability of being ham
                    
                    # Display prediction result
                    if prediction == 'spam':
                        st.markdown(f"""
                        <div class="prediction-box spam-prediction">
                            <h3>🚨 SPAM DETECTED</h3>
                            <p><strong>Confidence:</strong> {spam_prob:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box ham-prediction">
                            <h3>✅ LEGITIMATE MESSAGE</h3>
                            <p><strong>Confidence:</strong> {ham_prob:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.subheader("Confidence Breakdown")
                    col_spam, col_ham = st.columns(2)
                    
                    with col_spam:
                        st.metric("Spam Probability", f"{spam_prob:.1%}")
                        st.progress(spam_prob)
                    
                    with col_ham:
                        st.metric("Ham Probability", f"{ham_prob:.1%}")
                        st.progress(ham_prob)
                    
                    # Detailed analysis
                    with st.expander("🔬 Detailed Analysis"):
                        st.write(f"**Message Length:** {len(message)} characters")
                        st.write(f"**Word Count:** {len(message.split())} words")
                        st.write(f"**Spam Probability:** {spam_prob:.4f}")
                        st.write(f"**Ham Probability:** {ham_prob:.4f}")
                        
                        # Show top features that influenced the decision
                        st.write("**Key Features:**")
                        message_vectorized = vectorizer.transform([message])
                        feature_names = vectorizer.get_feature_names_out()
                        coefficients = model.coef_[0]
                        
                        # Get non-zero features for this message
                        non_zero_indices = message_vectorized.nonzero()[1]
                        if len(non_zero_indices) > 0:
                            feature_importance = []
                            for idx in non_zero_indices:
                                feature_name = feature_names[idx]
                                coefficient = coefficients[idx]
                                feature_importance.append((feature_name, coefficient))
                            
                            # Sort by absolute coefficient value
                            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Show top 10 features
                            for feature, coef in feature_importance[:10]:
                                color = "red" if coef > 0 else "green"
                                st.markdown(f"<span style='color: {color}'>{feature}: {coef:.3f}</span>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a message to check.")
    
    with col2:
        st.header("📊 Model Performance")
        
        # Performance metrics visualization
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        }
        
        fig = px.bar(
            x=metrics_data['Metric'], 
            y=metrics_data['Score'],
            title="Model Performance Metrics",
            color=metrics_data['Score'],
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataset distribution
        st.subheader("Dataset Distribution")
        spam_count = sum(df['Category'] == 'spam')
        ham_count = sum(df['Category'] == 'ham')
        
        fig_pie = px.pie(
            values=[ham_count, spam_count],
            names=['Ham', 'Spam'],
            title="Message Distribution",
            color_discrete_map={'Ham': '#4caf50', 'Spam': '#f44336'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Sample messages section
    st.header("📝 Sample Messages")
    
    # Show some sample messages
    sample_spam = df[df['Category'] == 'spam']['Message'].head(3).tolist()
    sample_ham = df[df['Category'] == 'ham']['Message'].head(3).tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Sample Spam Messages")
        for i, msg in enumerate(sample_spam, 1):
            with st.expander(f"Spam Example {i}"):
                st.write(msg)
                if st.button(f"Test Example {i}", key=f"spam_{i}"):
                    prediction, probability = predict_spam(msg, model, vectorizer)
                    spam_prob = probability[1]
                    st.write(f"**Prediction:** {prediction}")
                    st.write(f"**Spam Probability:** {spam_prob:.1%}")
    
    with col2:
        st.subheader("✅ Sample Ham Messages")
        for i, msg in enumerate(sample_ham, 1):
            with st.expander(f"Ham Example {i}"):
                st.write(msg)
                if st.button(f"Test Example {i}", key=f"ham_{i}"):
                    prediction, probability = predict_spam(msg, model, vectorizer)
                    spam_prob = probability[1]
                    st.write(f"**Prediction:** {prediction}")
                    st.write(f"**Spam Probability:** {spam_prob:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
        <p>This app uses a Logistic Regression model with TF-IDF vectorization to detect spam messages.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Could not load the model. Please make sure you have trained the model first by running 'make train'.")
    st.info("💡 To train the model, run the following command in your terminal: `make train`")
