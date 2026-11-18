import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_recall_curve, auc, f1_score, 
                             recall_score, precision_score, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_auc_score)
from features import clean

RANDOM_STATE = 29
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'initial_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_v1.joblib')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

def bootstrap_ci(y_true, y_proba, metric_func, n_bootstraps=1000):
    np.random.seed(RANDOM_STATE)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue
        scores.append(metric_func(y_true.iloc[indices], y_proba[indices]))
    return np.percentile(scores, [2.5, 97.5])

def main():
    print("1. Загрузка модели и теста.")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    df = clean(df)
    
    n = len(df)
    val_end = int(n * 0.85)
    df_test = df.iloc[val_end:].copy()
    
    y_test = (df_test['Anomaly_Label'] == 'Abnormal').astype(int)
    X_test = df_test.drop(columns=['Anomaly_Label'])
    
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"Размер теста: {len(df_test)} строк, из них аномалий: {y_test.sum()}")
    
    print("\n2. Метрики (порог 0.5):")
    pr_auc = pr_auc_score(y_test, y_proba)
    pr_ci = bootstrap_ci(y_test, y_proba, pr_auc_score)
    print(f"   PR-AUC: {pr_auc:.3f} (95% ДИ: {pr_ci[0]:.3f} - {pr_ci[1]:.3f})")
    
    print("\n3. Поиск оптимального порога (FN=10, FP=1):")
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = fn * 10 + fp * 1
        costs.append(cost)
        
    best_t = thresholds[np.argmin(costs)]
    print(f"   Оптимальный порог: {best_t:.3f}")
    
    y_pred_opt = (y_proba >= best_t).astype(int)
    print(f"   При пороге {best_t:.3f}: Recall={recall_score(y_test, y_pred_opt):.3f}, Precision={precision_score(y_test, y_pred_opt):.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_std = confusion_matrix(y_test, (y_proba >= 0.5).astype(int))
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    
    ConfusionMatrixDisplay(cm_std, display_labels=['Normal', 'Abnormal']).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Порог 0.5 (стандартный)')
    
    ConfusionMatrixDisplay(cm_opt, display_labels=['Normal', 'Abnormal']).plot(ax=axes[1], cmap='Oranges', colorbar=False)
    axes[1].set_title(f'Порог {best_t:.3f} (выбранный)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, marker='.', label=f'Random Forest (PR-AUC = {pr_auc:.3f})')
    plt.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Кривая «Точность-Полнота»')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'pr_curve.png'), dpi=300)
    plt.close()
    
    preprocessor = model.named_steps['prep']
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps['model'].feature_importances_
    clean_names = [name.split('__')[-1] for name in feature_names]
    
    imp_df = pd.DataFrame({'feature': clean_names, 'importance': importances})
    imp_df = imp_df.sort_values(by='importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=imp_df, palette='viridis')
    plt.title('Топ-10 важнейших признаков')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print("\nГрафики сохранены в папку reports/ (3 шт).")
    print("Готово.")

if __name__ == "__main__":
    main()