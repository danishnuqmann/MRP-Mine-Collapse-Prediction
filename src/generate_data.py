import pandas as pd
import numpy as np
import os

# --- 1. SMART PATHING (Matches Dashboard Logic) ---
# Finds the 'MRP' root folder automatically
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(CURRENT_SCRIPT_PATH)
ROOT_FOLDER = os.path.dirname(SRC_FOLDER)
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

# Create the folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"📁 Created folder at: {DATA_FOLDER}")

def generate_mine_dataset():
    samples = 300
    
    # 2. SAFE OPERATIONS DATA (Section 3.3.2)
    # Keeping values low and stable to represent normal conditions
    safe_data = {
        'Roof_Stress': np.random.normal(80, 3, samples),      # Normal pressure
        'Displacement': np.random.normal(0.2, 0.05, samples), # Minimal sagging
        'Vibration': np.random.normal(0.4, 0.1, samples),     # Normal background noise
        'Status': [1] * samples                               # 1 = Stable
    }
    
    # 3. FAULT / COLLAPSE DATA (Section 2.2)
    # Forcing a 'Progressive Failure' trend where values climb high
    fault_data = {
        'Roof_Stress': np.linspace(110, 200, samples) + np.random.normal(0, 5, samples),
        'Displacement': np.linspace(0.5, 3.0, samples) + np.random.normal(0, 0.1, samples),
        'Vibration': np.linspace(0.8, 2.5, samples) + np.random.normal(0, 0.2, samples),
        'Status': [0] * samples                               # 0 = Collapse Risk
    }
    
    # Convert to DataFrames
    df_safe = pd.DataFrame(safe_data)
    df_fault = pd.DataFrame(fault_data)
    
    # 4. SAVE THE FILES (Using Absolute Paths)
    # Individual files for the Dashboard simulation
    df_safe.to_csv(os.path.join(DATA_FOLDER, 'safe_data.csv'), index=False)
    df_fault.to_csv(os.path.join(DATA_FOLDER, 'fault_data.csv'), index=False)
    
    # Combined file for Model Training (Section 3.6)
    df_combined = pd.concat([df_safe, df_fault])
    df_combined.to_csv(os.path.join(DATA_FOLDER, 'mine_data.csv'), index=False)
    
    print("--- DATA GENERATION REPORT ---")
    print(f"✅ Safe Data saved: {len(df_safe)} rows")
    print(f"✅ Fault Data saved: {len(df_fault)} rows")
    print(f"✅ Total Training Data: {len(df_combined)} rows")
    print(f"📍 Location: {DATA_FOLDER}")

if __name__ == "__main__":
    generate_mine_dataset()