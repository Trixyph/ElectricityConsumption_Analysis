import streamlit as st
import pandas as pd
import sqlite3
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'smart_meter.db')

st.set_page_config(page_title="Мониторинг аномалий", layout="wide")

hide_deploy_style = """
        <style>
        [data-testid="stAppDeployButton"] {display: none !important;}
        
        #MainMenu {visibility: visible;}
        [data-testid="stToolbar"] {visibility: visible;}
        
        footer {visibility: hidden;}

        header {background: transparent !important;}
        </style>
        """
st.markdown(hide_deploy_style, unsafe_allow_html=True)

st.title("Система выявления аномалий энергопотребления")

if not os.path.exists(DB_PATH):
    st.error("База данных не найдена. Запустите process_batch.py")
    st.stop()

con = sqlite3.connect(DB_PATH)

hist = pd.read_sql("SELECT * FROM metrics_history", con)
if hist.empty:
    st.warning("Нет данных. Обработайте хотя бы одну порцию.")
    st.stop()

hist_pivot = hist.pivot(index="batch", columns="metric", values="value").reset_index()

latest = hist_pivot.iloc[-1]
prev = hist_pivot.iloc[-2] if len(hist_pivot) > 1 else latest

col1, col2, col3 = st.columns(3)
col1.metric("Доля аномалий (прогноз)", f"{latest['anomaly_share']:.1%}", 
            f"{(latest['anomaly_share'] - prev['anomaly_share'])*100:+.1f} п.п.")
col2.metric("Доля аномалий (факт)", f"{latest['actual_anomaly_share']:.1%}", 
            f"{(latest['actual_anomaly_share'] - prev['actual_anomaly_share'])*100:+.1f} п.п.")
col3.metric("PR-AUC на порции", f"{latest['pr_auc']:.3f}", 
            f"{(latest['pr_auc'] - prev['pr_auc']):+.3f}", delta_color="normal")

st.subheader("Предупреждения")
if latest['pr_auc'] < 0.65:
    st.error("PR-AUC упал ниже порога 0.65. Требуется переобучение.")
else:
    st.success("Качество модели в норме.")

if latest['anomaly_share'] > latest['actual_anomaly_share'] * 1.5:
    st.warning("Доля предсказанных аномалий значительно выше фактической.")

st.subheader("Динамика показателей по порциям")
st.line_chart(hist_pivot.set_index("batch")[['anomaly_share', 'actual_anomaly_share']])

st.subheader(f"Интервалы, требующие проверки (Порция: {latest['batch']})")
query = f"""
    SELECT p.timestamp, p.probability, r.electricity_consumed 
    FROM predictions p
    JOIN raw_data r ON p.batch = r.batch AND p.timestamp = r.timestamp
    WHERE p.batch = '{latest['batch']}' AND p.is_anomaly = 1
    ORDER BY p.probability DESC
    LIMIT 10
"""
suspects = pd.read_sql(query, con)
st.dataframe(suspects, use_container_width=True)

con.close()