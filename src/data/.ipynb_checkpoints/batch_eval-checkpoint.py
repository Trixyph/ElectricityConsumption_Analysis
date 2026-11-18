import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.metrics import precision_recall_curve, auc, recall_score
from features import clean
from quality import check_quality

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'initial_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_v1.joblib')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
INCOMING_DIR = os.path.join(BASE_DIR, 'data', 'incoming')

def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min_val, max_val):
        input += -(np.min(input))
        input /= np.max(input) / (max_val - min_val)
        input += min_val
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

def main():
    print("1. Загрузка модели.")
    model = joblib.load(MODEL_PATH)
    
    df_train = clean(pd.read_csv(DATA_PATH))
    train_end = int(len(df_train) * 0.70)
    expected_consumed = df_train.iloc[:train_end]['Electricity_Consumed'].values
    
    BATCH_THRESHOLD = 0.139
    
    print("\n2. Обработка новых порций.")
    batches = sorted(glob.glob(os.path.join(INCOMING_DIR, 'batch_*.csv')))
    
    if not batches:
        print("Внимание: В папке data/incoming нет файлов batch_*.csv")
        return

    for batch_path in batches:
        filename = os.path.basename(batch_path)
        df_batch = pd.read_csv(batch_path)
        
        # Проверка качества
        q_report = check_quality(df_batch)
        q_status = "OK" if q_report["ok"] else "ОШИБКИ"
        
        if not q_report["ok"]:
            print(f"\n{filename}")
            print(f"Проверка качества: {q_status} (Отбраковано)")
            continue

        df_clean = clean(df_batch)
        y_true = (df_clean['Anomaly_Label'] == 'Abnormal').astype(int)
        X = df_clean.drop(columns=['Anomaly_Label'], errors='ignore')
        
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= BATCH_THRESHOLD).astype(int)
        
        pr_auc = pr_auc_score(y_true, y_proba)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        actual_consumed = df_clean['Electricity_Consumed'].values
        psi_val = calculate_psi(expected_consumed, actual_consumed)
        
        print(f"\n{filename}")
        print(f"Проверка качества: {q_status} (Строк: {len(df_batch)})")
        print(f"Аномалий в порции: {y_true.sum()}")
        print(f"PR-AUC: {pr_auc:.3f}")
        print(f"Recall (порог {BATCH_THRESHOLD}): {recall:.3f}")
        print(f"PSI (Electricity_Consumed): {psi_val:.3f}")

if __name__ == "__main__":
    main()