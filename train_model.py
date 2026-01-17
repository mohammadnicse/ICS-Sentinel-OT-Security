import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_swat_model():
    print("Loading Processed Training Data...")
    
    try:
        df = pd.read_pickle("data/processed/train_data.pkl")
    except FileNotFoundError:
        print("ERROR: 'train_data.pkl' not found. Run 'process_swat.py' first.")
        return
    
    # SCALING
    # Standardize data because Flow (0-5) and Level (0-1000) have different scales
    print("   Scaling 51 sensor features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # TRAINING
    # Isolation Forest: An unsupervised algorithm that identifies anomalies
    # n_jobs=-1 uses all CPU cores for speed (Hardware/Performance optimization)
    print("Training AI Model (Isolation Forest)...")
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    
    # SAVING
    print("Saving Model Artifacts to 'models/'...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/swat_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("SUCCESS: Model trained and saved.")

if __name__ == "__main__":
    train_swat_model()