import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc

from storage import get_connection, init_db
from quality import check_quality
from features import clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_v1.joblib')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

THRESHOLD = 0.139
MODEL_VERSION = "v1"

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

def process_batch(filepath):
    batch_name = os.path.basename(filepath).replace('.csv', '')
    init_db()
    con = get_connection()
    
    start_time = datetime.now()
    df_raw = pd.read_csv(filepath)
    
    q_report = check_quality(df_raw)
    if not q_report["ok"]:
        con.execute("INSERT INTO runs (batch, started_at, status, rows, message) VALUES (?, ?, ?, ?, ?)",
                    (batch_name, start_time.isoformat(), "rejected", len(df_raw), "Ошибка качества данных"))
        con.commit()
        return {"status": "rejected", "reason": "quality_check_failed"}
        
    df_clean = clean(df_raw)
    model = joblib.load(MODEL_PATH)
    
    X = df_clean.drop(columns=['Anomaly_Label'], errors='ignore')
    y_true = (df_clean['Anomaly_Label'] == 'Abnormal').astype(int)
    
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    
    raw_records = df_clean[['Timestamp', 'Electricity_Consumed', 'Temperature', 
                            'Humidity', 'Wind_Speed', 'Avg_Past_Consumption', 'Anomaly_Label']].values.tolist()
    for r in raw_records:
        con.execute("INSERT OR REPLACE INTO raw_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (batch_name, str(r[0]), r[1], r[2], r[3], r[4], r[5], r[6]))
                    
    pred_records = [(batch_name, str(ts), float(prob), int(pred), MODEL_VERSION) 
                    for ts, prob, pred in zip(df_clean['Timestamp'], y_proba, y_pred)]
    con.executemany("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?)", pred_records)
    
    metrics = {
        'anomaly_share': float(y_pred.mean()),
        'actual_anomaly_share': float(y_true.mean()),
        'pr_auc': float(pr_auc_score(y_true, y_proba))
    }
    
    for m_name, m_val in metrics.items():
        con.execute("INSERT OR REPLACE INTO metrics_history VALUES (?, ?, ?, ?)",
                    (batch_name, start_time.isoformat(), m_name, m_val))
                    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    con.execute("INSERT INTO runs (batch, started_at, status, rows, message) VALUES (?, ?, ?, ?, ?)",
                (batch_name, start_time.isoformat(), "ok", len(df_clean), f"Время: {duration:.2f}с"))
    con.commit()
    con.close()
    
    return {"status": "ok", "batch": batch_name, "metrics": metrics}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python process_batch.py <путь_к_файлу.csv>")
    else:
        result = process_batch(sys.argv[1])
        print(json.dumps(result, indent=2, ensure_ascii=False))