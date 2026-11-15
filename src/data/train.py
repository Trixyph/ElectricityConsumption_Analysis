import pandas as pd
import numpy as np
import joblib
import json
import time
import os
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from features import clean, make_pipeline

RANDOM_STATE = 29

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'initial_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

def evaluate_model(name, pipeline, X, y, cv):
    pr_auc_scores, f1_scores, rec_scores = [], [], []
    start_time = time.time()
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = pipeline.predict(X_val)
        
        pr_auc_scores.append(pr_auc_score(y_val, y_proba))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        rec_scores.append(recall_score(y_val, y_pred, zero_division=0))
        
    train_time = time.time() - start_time
    
    return {
        'model': name,
        'pr_auc_mean': np.mean(pr_auc_scores),
        'pr_auc_std': np.std(pr_auc_scores),
        'f1_mean': np.mean(f1_scores),
        'recall_mean': np.mean(rec_scores),
        'train_time_sec': round(train_time, 2)
    }

def main():
    print("1. Загрузка данных.")
    df = pd.read_csv(DATA_PATH)
    df = clean(df)
    
    y = (df['Anomaly_Label'] == 'Abnormal').astype(int)
    X = df.drop(columns=['Anomaly_Label'])
    
    n = len(df)
    train_end = int(n * 0.70)
    # val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    # X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end] (пока не используем)
    # X_test, y_test = X.iloc[val_end:], y.iloc[val_end:] (пока не используем)
    
    print(f"Размер Train: {len(X_train)} строк, из них аномалий: {y_train.sum()}")
    
    tscv = TimeSeriesSplit(n_splits=5, gap=48)
    
    print("2. Инициализация.")
    
    models = {
        'Dummy (Most Frequent)': DummyClassifier(strategy='prior'),
        'Logistic Regression L2': LogisticRegression(C=1.0, penalty='l2', max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'Logistic Regression L1': LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'kNN (k=25)': KNeighborsClassifier(n_neighbors=25),
        'Decision Tree (max_depth=5)': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=RANDOM_STATE)
    }
    
    results = []
    
    print("3. Запуск кросс-валидации.")
    for name, model in models.items():
        if name == 'Dummy (Most Frequent)':
            res = {'model': name, 'pr_auc_mean': round(y_train.mean(), 3), 'pr_auc_std': 0.0, 'f1_mean': 0.0, 'recall_mean': 0.0, 'train_time_sec': 0.0}
        else:
            pipeline = make_pipeline(model)
            res = evaluate_model(name, pipeline, X_train, y_train, tscv)
        
        results.append(res)
        print(f"   [{name}] PR-AUC: {res['pr_auc_mean']:.3f} ± {res['pr_auc_std']:.3f}, Время: {res['train_time_sec']}с")
    
    print("4. Подбор гиперпараметров для Gradient Boosting.")
    gb_pipeline = make_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    param_grid = {
        'model__learning_rate': [0.05, 0.1],
        'model__max_iter': [100, 200],
        'model__max_depth': [3, 5]
    }
    
    grid_search = GridSearchCV(gb_pipeline, param_grid, cv=tscv, scoring='average_precision', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"   Лучшие параметры: {grid_search.best_params_}")
    print(f"   Лучший PR-AUC: {grid_search.best_score_:.3f}")
    
    results.append({
        'model': 'Gradient Boosting (Tuned)',
        'pr_auc_mean': grid_search.best_score_,
        'pr_auc_std': 0.0,
        'f1_mean': 0.0, 
        'recall_mean': 0.0,
        'train_time_sec': 0.0
    })
    
    print("5. Сохранение результатов...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORT_DIR, 'experiments.csv'), index=False)
    
    best_row = results_df[results_df['model'] != 'Dummy (Most Frequent)'].sort_values('pr_auc_mean', ascending=False).iloc[0]
    print(f"\nЛучшая модель: {best_row['model']} (PR-AUC: {best_row['pr_auc_mean']:.3f})")
    
    if 'Tuned' in best_row['model']:
        final_model = grid_search.best_estimator_
    else:
        final_model = make_pipeline(models[best_row['model']])
    
    print("Обучение финальной модели на Train.")
    final_model.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, 'model_v1.joblib')
    joblib.dump(final_model, model_path)
    
    info = {
        "version": "v1",
        "trained": datetime.now().strftime("%Y-%m-%d"),
        "data": "initial_data.csv (train subset)",
        "algorithm": best_row['model'],
        "cv_pr_auc": round(best_row['pr_auc_mean'], 3),
        "random_state": RANDOM_STATE
    }
    with open(os.path.join(MODEL_DIR, 'model_v1.json'), 'w') as f:
        json.dump(info, f, indent=1)
        
    print("Модель и метрики сохранены.")

if __name__ == "__main__":
    main()