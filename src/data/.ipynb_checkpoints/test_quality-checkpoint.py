import pandas as pd
from quality import check_quality
import os

df_bad = pd.read_csv('../data/incoming/batch_01.csv')

df_bad.loc[5:10, 'Timestamp'] = None

df_bad.loc[15:20, 'Electricity_Consumed'] = 1.5

df_bad = df_bad.head(100)

print("Проверка намеренно испорченного файла")
report = check_quality(df_bad)
print(f"Порция прошла проверку? {report['ok']}")
print(f"Количество строк: {report['rows']}")
for check, res in report['checks'].items():
    if res['violations'] > 0:
        print(f"  [ОШИБКА] {check}: найдено нарушений {res['violations']}")