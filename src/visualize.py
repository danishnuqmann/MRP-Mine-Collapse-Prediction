import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # Load both datasets
    safe_df = pd.read_csv(os.path.join("data", "safe_data.csv"))
    fault_df = pd.read_csv(os.path.join("data", "fault_data.csv"))
    
    # Create a 2-column, 3-row plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    plt.suptitle('Comparison Analysis: Safe Mode vs. Fault Mode (Early Warning)', fontsize=18)

    # --- Column 1: SAFE MODE (The "Baseline") ---
    axes[0, 0].plot(safe_df['Roof_Stress'], color='green')
    axes[0, 0].set_title('Safe: Roof Stress (Steady)')
    axes[0, 0].set_ylabel('MPa')
    
    axes[1, 0].plot(safe_df['Displacement'], color='green')
    axes[1, 0].set_title('Safe: Displacement (No Movement)')
    axes[1, 0].set_ylabel('mm')
    
    axes[2, 0].plot(safe_df['Vibration'], color='green')
    axes[2, 0].set_title('Safe: Vibration (Quiet)')
    axes[2, 0].set_xlabel('Time Steps')

    # --- Column 2: FAULT MODE (The "Early Warning") ---
    axes[0, 1].plot(fault_df['Roof_Stress'], color='red')
    axes[0, 1].set_title('FAULT: Roof Stress (Rising!)')
    
    axes[1, 1].plot(fault_df['Displacement'], color='red')
    axes[1, 1].set_title('FAULT: Displacement (Sagging!)')
    
    axes[2, 1].plot(fault_df['Vibration'], color='red')
    axes[2, 1].set_title('FAULT: Vibration (Anomalous!)')
    axes[2, 1].set_xlabel('Time Steps')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('data/mode_comparison.png')
    print("Babe, the comparison chart is saved to data/mode_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_comparison()