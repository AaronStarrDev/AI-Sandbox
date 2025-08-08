#import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processing import process_log_data

def train_classifier():
    # For simplicity, use processed logs as the dataset
    # Assume we have logs processed from files, with label (0 for normal, 1 for anomaly)
    
    # Feature vectors (X) and target labels (y)
    Xvectors = [
        [0, 1],  # Normal log, no missing values
        [1, 0],  # Missing event_type
        [0, 1],  # Normal log
        [1, 0]   # Missing event_type
    ]
    Yvectors = [0, 1, 0, 1]  # 0: Normal, 1: Anomalous
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(Xvectors, Yvectors, test_size=0.2, random_state=42)
    
    # Initialize Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Using the classifier on processed log data
model = train_classifier()

# Process the logs from files
log_1_features = process_log_data("AI-Sandbox/TestEvents/log_test_1.json", 1)
log_2_features = process_log_data("AI-Sandbox/TestEvents/log_test_2.json", 2)

# Predict anomalies using the trained model
prediction_1 = model.predict([list(log_1_features.values())])
prediction_2 = model.predict([list(log_2_features.values())])

print(f"Prediction for Log 1: {prediction_1}")
print(f"Prediction for Log 2: {prediction_2}")
