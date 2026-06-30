import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --- BULLETPROOF PATH CONFIGURATION ---
# 1. Get the directory where train.py is currently located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the main project folder
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# 3. Create absolute paths to datasets and models
DATASETS_DIR = os.path.join(PARENT_DIR, 'datasets')
MODELS_DIR = os.path.join(PARENT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

CONFIG = {
    'heart': {'file': 'heart.csv', 'target': 'target'},
    'diabetes': {'file': 'diabetes.csv', 'target': 'Outcome'},
    'breast-cancer': {'file': 'breast-cancer.csv', 'target': 'diagnosis'}
}

def train_and_evaluate(name, data_info):
    print(f"--- Training models for {name} ---")
    filepath = os.path.join(DATASETS_DIR, data_info['file'])
    
    if not os.path.exists(filepath):
        # Fallback just in case your folder is still named 'dataset' (singular)
        fallback_path = os.path.join(PARENT_DIR, 'dataset', data_info['file'])
        if os.path.exists(fallback_path):
            filepath = fallback_path
        else:
            print(f"❌ Error: Dataset not found at {filepath}")
            return

    df = pd.read_csv(filepath)
    
    # Preprocessing
    df.fillna(df.median(numeric_only=True), inplace=True)
    target_col = data_info['target']
    
    # Bulletproof target encoding (Converts 'B'/'M' to 0/1 safely)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    X = df.drop(columns=[target_col])
    if 'id' in X.columns:
        X = X.drop(columns=['id'])
        
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_f1 = -1
    best_name = ""
    best_metrics = {}

    for m_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc = roc_auc_score(y_test, y_proba)
        except:
            roc = 0.0
            
        cm = confusion_matrix(y_test, y_pred).tolist()

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = m_name
            best_metrics = {
                'algorithm': best_name,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc,
                'confusion_matrix': cm,
                'features': list(X.columns)
            }

    # Save artifacts
    prefix = name.split('-')[0]
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{prefix}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl"))
    with open(os.path.join(MODELS_DIR, f"{prefix}_metrics.json"), 'w') as f:
        json.dump(best_metrics, f, indent=4)
        
    print(f"✅ Best model for {name}: {best_name} (F1: {best_f1:.4f}) saved successfully!")

if __name__ == "__main__":
    for disease, info in CONFIG.items():
        train_and_evaluate(disease, info)
    print("🎉 All training complete.")