import pandas as pd
import numpy as np
import os

def clean_swat_data():
    print("Checking directories...")
    # to make sure the directories exist
    os.makedirs("data/processed", exist_ok=True)

    print("Loading SWaT Raw Datasets... (This requires RAM)")
    
    # LOAD RAW DATA
    # to make sure you have put the CSVs in 'data/raw/'
    try:
        normal_df = pd.read_csv("data/raw/SWaT_Dataset_Normal_v1.csv", low_memory=False)
        attack_df = pd.read_csv("data/raw/SWaT_Dataset_Attack_v0.csv", low_memory=False)
    except FileNotFoundError:
        print("ERROR: Files not found. Please ensure 'SWaT_Dataset_Normal_v1.csv' and 'SWaT_Dataset_Attack_v0.csv' are inside 'data/raw/' folder.")
        return

    print(f"   Loaded: Normal ({len(normal_df)} rows), Attack ({len(attack_df)} rows)")

    # CLEAN COLUMN NAMES
    # Strip whitespace from " FIT 101 " -> "FIT101"
    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()
    
    # HANDLE LABELS (Convert Text to Numbers)
    # The column is usually named 'Normal/Attack'
    target_col = 'Normal/Attack'
    
    if target_col in attack_df.columns:
        # Map 'Normal' -> 0, 'Attack' -> 1
        # 'A ttack' is a typo found in the original SWaT dataset, we fix it here.
        attack_df[target_col] = attack_df[target_col].map({'Normal': 0, 'Attack': 1, 'A ttack': 1})
    
    # FEATURE SELECTION
    # We remove Timestamp and Labels for the Training set (Unsupervised learning uses only physics data)
    features = list(normal_df.columns)
    if 'Timestamp' in features: features.remove('Timestamp')
    if 'Normal/Attack' in features: features.remove('Normal/Attack')
    
    print(f"   Selected {len(features)} physical sensors for training.")

    # SAVE PROCESSED DATA
    print("Saving processed data to pickles (Fast Load)...")
    
    # Save training data (features only)
    normal_df[features].to_pickle("data/processed/train_data.pkl")
    
    # Save testing data (keep Timestamp and Labels for validation)
    attack_df.to_pickle("data/processed/test_data.pkl")
    
    print("COMPLETE. Data is ready in 'data/processed/'.")

if __name__ == "__main__":
    clean_swat_data()