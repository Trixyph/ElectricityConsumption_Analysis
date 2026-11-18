import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'data', 'smart_meter.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as con:
        cursor = con.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                batch TEXT, timestamp TEXT, electricity_consumed REAL,
                temperature REAL, humidity REAL, wind_speed REAL,
                avg_past_consumption REAL, anomaly_label TEXT,
                PRIMARY KEY (batch, timestamp)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                batch TEXT, timestamp TEXT, probability REAL, 
                is_anomaly INTEGER, model_version TEXT,
                PRIMARY KEY (batch, timestamp)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_history (
                batch TEXT, processed_at TEXT, metric TEXT, value REAL,
                PRIMARY KEY (batch, metric)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT, batch TEXT,
                started_at TEXT, status TEXT, rows INTEGER, message TEXT
            )
        ''')
        con.commit()
    print("База данных инициализирована.")

if __name__ == "__main__":
    init_db()