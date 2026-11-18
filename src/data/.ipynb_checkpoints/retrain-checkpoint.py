import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
from features import clean, make_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'initial_data.csv')
INCOMING_DIR = os.path.join(BASE_DIR, 'data', 'incoming')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RANDOM_STATE = 29

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

def retrain():
    print("Загрузка накопленных данных.")
    df_old = pd.read_csv(DATA_PATH)
    batches = [pd.read_csv(os.path.join(INCOMING_DIR, f)) for f in os.listdir(INCOMING_DIR) if f.endswith('.csv')]
    df_all = pd.concat([df_old] + batches, ignore_index=True)
    df_all = clean(df_all)
    
    y = (df_all['Anomaly_Label'] == 'Abnormal').astype(int)
    X = df_all.drop(columns=['Anomaly_Label'])
    
    split_idx = int(len(df_all) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    print("Обучение претендента (v2).")
    model_v2 = make_pipeline(RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1))
    model_v2.fit(X_train, y_train)
    
    y_proba = model_v2.predict_proba(X_test)[:, 1]
    pr_auc = pr_auc_score(y_test, y_proba)
    print(f"PR-AUC претендента: {pr_auc:.3f}")
    
    model_path = os.path.join(MODEL_DIR, 'model_v2.joblib')
    joblib.dump(model_v2, model_path)
    
    info = {
        "version": "v2",
        "trained": datetime.now().strftime("%Y-%m-%d"),
        "algorithm": "RandomForest (200 trees)",
        "cv_pr_auc": round(pr_auc, 3),
        "previous_version": "v1"
    }
    with open(os.path.join(MODEL_DIR, 'model_v2.json'), 'w') as f:
        json.dump(info, f, indent=1)
        
    print("Модель v2 сохранена. Указатель current.json обновлен (имитация).")

if __name__ == "__main__":
    retrain()