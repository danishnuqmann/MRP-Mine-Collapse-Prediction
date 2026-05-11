import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from main import prepare_data, build_model # Using your existing brain

def run_comprehensive_evaluation():
    # 1. Load the Combined Training Data
    print("Loading data for evaluation...")
    X, y = prepare_data("mine_data.csv")
    
    # Split for testing (20% for the exam)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build and Train
    model = build_model((X.shape[1], X.shape[2]))
    print("Training model for performance metrics...")
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # 3. Generate Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 4. RESULTS: CLASSIFICATION REPORT
    # This gives you Precision, Recall, F1-Score, and Support
    print("\n" + "="*30)
    print(" FYP 2 PERFORMANCE METRICS ")
    print("="*30)
    print(classification_report(y_test, y_pred, target_names=['Collapse (Fault)', 'Stable']))

    # 5. RESULTS: CONFUSION MATRIX (TP, TN, FP, FN)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- Confusion Matrix Breakdown ---")
    print(f"True Positives (TP): {tp}  (Correctly predicted Stable)")
    print(f"True Negatives (TN): {tn}  (Correctly predicted Collapse)")
    print(f"False Positives (FP): {fp}  (False Alarm - Safe but called Danger)")
    print(f"False Negatives (FN): {fn}  (DANGEROUS - Collapse but called Safe)")
    
    # 6. VISUALIZATION for your Report
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Collapse', 'Stable'], 
                yticklabels=['Collapse', 'Stable'])
    plt.title('Confusion Matrix: Mine Roof Prediction')
    plt.ylabel('Actual Condition')
    plt.xlabel('Predicted Condition')
    plt.savefig('data/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    run_comprehensive_evaluation()