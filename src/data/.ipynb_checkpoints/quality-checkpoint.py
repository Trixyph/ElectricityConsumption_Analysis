import pandas as pd

def check_quality(df: pd.DataFrame) -> dict:
    checks = {
        "Q1 Timestamp уникален и без пропусков": (
            int(df['Timestamp'].isna().sum() + df['Timestamp'].duplicated().sum()), 
            True
        ),
        "Q2 Electricity_Consumed в диапазоне [0, 1]": (
            int((~df['Electricity_Consumed'].between(0, 1)).sum() + df['Electricity_Consumed'].isna().sum()), 
            True
        ),
        "Q3 Погодные признаки в диапазоне [0, 1]": (
            int((~df['Temperature'].between(0, 1)).sum() + (~df['Humidity'].between(0, 1)).sum() + (~df['Wind_Speed'].between(0, 1)).sum()), 
            False
        ),
        "Q4 Avg_Past_Consumption в диапазоне [0, 1]": (
            int((~df['Avg_Past_Consumption'].between(0, 1)).sum()), 
            True
        ),
        "Q5 Объем порции не менее 400 строк": (
            int(len(df) < 400), 
            False
        )
    }
    
    report = {k: {"violations": v, "blocking": b} for k, (v, b) in checks.items()}
    ok = not any(r["violations"] and r["blocking"] for r in report.values())
    
    return {"ok": ok, "rows": len(df), "checks": report}