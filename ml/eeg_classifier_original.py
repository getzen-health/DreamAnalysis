import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal

# 1. Simulate neural signal data (mimicking EEG or Neuralink-like data)
def generate_neural_signals(n_samples=1000, n_features=64, n_classes=3):
    """
    Generate synthetic neural signal data with specified number of samples, features, and classes.
    Each class represents a different brain state (e.g., rest, movement intention, visual stimulus).
    """
    np.random.seed(42)
    time = np.linspace(0, 1, n_features)
    data = []
    labels = []
    
    for _ in range(n_samples):
        # Simulate different brain states with varying frequency components
        if np.random.rand() < 0.33:
            # Class 0: Rest state (low-frequency noise)
            sig = np.random.normal(0, 0.5, n_features) + 0.2 * np.sin(2 * np.pi * 5 * time)
            labels.append(0)
        elif np.random.rand() < 0.66:
            # Class 1: Movement intention (alpha/beta band activity)
            sig = np.random.normal(0, 0.5, n_features) + 0.5 * np.sin(2 * np.pi * 10 * time)
            labels.append(1)
        else:
            # Class 2: Visual stimulus (higher frequency gamma band)
            sig = np.random.normal(0, 0.5, n_features) + 0.3 * np.sin(2 * np.pi * 30 * time)
            labels.append(2)
        data.append(sig)
    
    return np.array(data), np.array(labels)

# 2. Preprocess the data
def preprocess_data(X, y):
    """
    Split data into training and testing sets, and standardize features.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# 3. Train AI model
def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier to classify neural signals.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rest', 'Movement', 'Visual']))

# 5. Visualize sample neural signals
def visualize_signals(X, y, n_samples=3):
    """
    Plot sample neural signals for each class.
    """
    plt.figure(figsize=(10, 6))
    time = np.linspace(0, 1, X.shape[1])
    classes = ['Rest', 'Movement', 'Visual']
    
    for i in range(n_samples):
        idx = np.where(y == i)[0][0]  # Get one sample per class
        plt.plot(time, X[idx], label=f'{classes[i]} Signal')
    
    plt.title('Sample Neural Signals by Class')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate synthetic neural data
    X, y = generate_neural_signals(n_samples=1000, n_features=64, n_classes=3)
    
    # Visualize sample signals
    visualize_signals(X, y)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Save the model (optional)
    import joblib
    joblib.dump(model, 'neural_signal_classifier.pkl')