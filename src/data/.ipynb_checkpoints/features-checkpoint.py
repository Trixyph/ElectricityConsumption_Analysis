import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

NUM_COLS = [
    "Electricity_Consumed", "Temperature", "Humidity", "Wind_Speed",
    "Avg_Past_Consumption", "lag_1", "rolling_mean_6", "diff_avg"
]
CAT_COLS = ["part_of_day", "is_night"]

def clean(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop_duplicates(subset=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

def add_features(df):
    df = df.copy()
    
    df['hour'] = df['Timestamp'].dt.hour
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    
    bins = [-1, 5, 11, 17, 21, 24]
    labels = ['night', 'morning', 'day', 'evening', 'night_late']
    df['part_of_day'] = pd.cut(df['hour'], bins=bins, labels=labels).astype(str)
    
    df['lag_1'] = df['Electricity_Consumed'].shift(1)
    
    df['rolling_mean_6'] = df['Electricity_Consumed'].rolling(window=6, min_periods=1).mean()
    
    df['diff_avg'] = df['Electricity_Consumed'] - df['Avg_Past_Consumption']
    
    df['lag_1'] = df['lag_1'].fillna(df['Avg_Past_Consumption'])
    
    return df

def make_pipeline(model):
    numeric_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COLS),
            ('cat', categorical_transformer, CAT_COLS)
        ])

    return Pipeline(steps=[
        ('features', FunctionTransformer(add_features)),
        ('prep', preprocessor),
        ('model', model)
    ])