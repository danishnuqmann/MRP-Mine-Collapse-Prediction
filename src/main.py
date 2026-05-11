import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import keras
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# 1. DATA LOADING FUNCTION
def prepare_data(filename):
    df = pd.read_csv(os.path.join("data", filename))
    # Parameters: Stress, Displacement, Vibration
    features = df[['Roof_Stress', 'Displacement', 'Vibration']].values
    labels = df['Status'].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(10, len(scaled_features)):
        X.append(scaled_features[i-10:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

# 2. HYBRID CNN-LSTM ARCHITECTURE
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape), # CNN for spikes
        layers.MaxPooling1D(2),
        layers.LSTM(50), # LSTM for history
        layers.Dense(1, activation='sigmoid') # Final decision
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. RUNNING THE SYSTEM
if __name__ == "__main__":
    # A. Train on the combined data
    print("Step 1: Teaching the brain using 'mine_data.csv'...")
    X_train, y_train = prepare_data("mine_data.csv")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, verbose=1)
    
    # B. Test the "Safe Mode"
    print("\nStep 2: Testing 'Safe Mode' (Expecting Stable)...")
    X_safe, _ = prepare_data("safe_data.csv")
    safe_pred = model.predict(X_safe[-1:]) # Check latest reading
    status = "✅ STABLE" if safe_pred[0][0] > 0.5 else "⚠️ ALERT"
    print(f"Result: {status} (Prob: {safe_pred[0][0]:.2f})")

    # C. Test the "Early Warning" (Fault Mode)
    print("\nStep 3: Testing 'Early Warning' (Expecting Danger)...")
    X_fault, _ = prepare_data("fault_data.csv")
    fault_pred = model.predict(X_fault[-1:]) # Check latest reading
    status = "✅ STABLE" if fault_pred[0][0] > 0.5 else "⚠️ ALERT"
    print(f"Result: {status} (Prob: {fault_pred[0][0]:.2f})")