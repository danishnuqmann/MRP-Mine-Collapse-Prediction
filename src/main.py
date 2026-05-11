import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    # Features: Stress, Displacement, Vibration
    X_raw = df[['Roof_Stress', 'Displacement', 'Vibration']].values
    y_raw = df['Status'].values

    # Scaling (Section 3.4.2)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Creating Sequences for LSTM (Window size = 10)
    X, y = [], []
    for i in range(10, len(X_scaled)):
        X.append(X_scaled[i-10:i])
        y.append(y_raw[i])
    
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        # CNN Layers (Spatial Feature Extraction)
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM Layers (Temporal Learning)
        LSTM(50, activation='relu', return_sequences=False),
        
        # Output Layer (Binary Classification: 1=Stable, 0=Critical)
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model_full(model, X_test, y_test):
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate standard metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract Confusion Matrix (TP, TN, FP, FN)
    # Note: 1 = Stable (Positive), 0 = Critical (Negative)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        "Accuracy": report['accuracy'],
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1": report['1']['f1-score'],
        "Support": report['1']['support'],
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }
    return metrics