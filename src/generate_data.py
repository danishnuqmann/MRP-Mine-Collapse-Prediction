import pandas as pd
import numpy as np
import os

# --- 1. SMART PATHING ---
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(CURRENT_SCRIPT_PATH)
ROOT_FOLDER = os.path.dirname(SRC_FOLDER)
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def generate_mine_dataset():
    samples = 500 # Increased for better learning
    
    # 2. SAFE OPERATIONS (Very Stable, Low Values)
    safe_data = {
        'Roof_Stress': np.random.normal(50, 2, samples),      # Low pressure
        'Displacement': np.random.normal(0.1, 0.02, samples), # No movement
        'Vibration': np.random.normal(0.2, 0.05, samples),    # Quiet
        'Status': [1] * samples                               # 1 = STABLE
    }
    
    # 3. FAULT / COLLAPSE (High Values, Fast Growth)
    fault_data = {
        'Roof_Stress': np.linspace(150, 250, samples) + np.random.normal(0, 5, samples),
        'Displacement': np.linspace(1.5, 5.0, samples) + np.random.normal(0, 0.2, samples),
        'Vibration': np.linspace(1.0, 4.0, samples) + np.random.normal(0, 0.3, samples),
        'Status': [0] * samples                               # 0 = CRITICAL
    }
    
    df_safe = pd.DataFrame(safe_data)
    df_fault = pd.DataFrame(fault_data)
    
    # Save files
    df_safe.to_csv(os.path.join(DATA_FOLDER, 'safe_data.csv'), index=False)
    df_fault.to_csv(os.path.join(DATA_FOLDER, 'fault_data.csv'), index=False)
    pd.concat([df_safe, df_fault]).to_csv(os.path.join(DATA_FOLDER, 'mine_data.csv'), index=False)
    
    print(f"✅ Clean data generated in: {DATA_FOLDER}")

if __name__ == "__main__":
    generate_mine_dataset()