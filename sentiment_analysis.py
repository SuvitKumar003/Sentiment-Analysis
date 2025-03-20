import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load and preprocess the data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Assuming the format is: text\tlabel
            text, label = line.strip().split('\t')
            data.append({'text': text, 'label': int(label)})
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the text data"""
    # Convert labels to binary (assuming 0 is negative, 1 is positive)
    df['label'] = df['label'].map({0: 0, 1: 1})
    return df

def train_model(X_train, y_train):
    """Train the sentiment analysis model"""
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Initialize and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate the model and create confusion matrix"""
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load training data
    print("Loading training data...")
    train_df = load_data('Dataset/training.txt')
    train_df = preprocess_data(train_df)
    
    # Split data into features and labels
    X = train_df['text']
    y = train_df['label']
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training the model...")
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    evaluate_model(model, vectorizer, X_val, y_val)
    
    # Load and evaluate on test data
    print("\nLoading and evaluating on test data...")
    test_df = load_data('Dataset/testdata.txt')
    test_df = preprocess_data(test_df)
    evaluate_model(model, vectorizer, test_df['text'], test_df['label'])

if __name__ == "__main__":
    main() 