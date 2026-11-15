import pandas as pd
import os

def split_dataset():
    # Пути
    input_file = 'data/raw/smart_meter_data.csv'
    output_dir = 'data/incoming'
    initial_file = 'data/raw/initial_data.csv'

    # Создаем папку для порций, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем данные
    print("Загрузка данных...")
    df = pd.read_csv(input_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # Вычисляем индекс для 60%
    split_idx = int(len(df) * 0.6)
    
    # Исходная часть (60%)
    initial_df = df.iloc[:split_idx].copy()
    initial_df.to_csv(initial_file, index=False)
    print(f"Исходная часть сохранена: {initial_file} ({len(initial_df)} строк)")

    # Оставшиеся 40% делим на 4 порции (по 10% каждая)
    incoming_df = df.iloc[split_idx:].copy()
    batch_size = len(incoming_df) // 4

    for i in range(4):
        start_idx = i * batch_size
        # Для последней порции берем все остатки
        if i == 3:
            end_idx = len(incoming_df)
        else:
            end_idx = start_idx + batch_size
            
        batch = incoming_df.iloc[start_idx:end_idx].copy()
        batch_file = os.path.join(output_dir, f'batch_{i+1:02d}.csv')
        batch.to_csv(batch_file, index=False)
        print(f"Порция сохранена: {batch_file} ({len(batch)} строк)")

    print("Разделение завершено успешно!")

if __name__ == "__main__":
    split_dataset()