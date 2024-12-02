# scripts/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    # Handle missing values
    # print(f"Number of missing data \n{data.isnull().sum()}")
    # Drop columns with no valuable information
    if 'snow' in data.columns and data['snow'].nunique() == 1:
        data = data.drop(columns=['snow'])

    return data


def process_time_data_scale(data):
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    data['hour_of_day_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
    data['hour_of_day_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)

    data = data.drop(columns=['month', 'day_of_week', 'hour_of_day', ], axis=1)
    # numeric_features = data.drop(['weekday', 'holiday'], axis=1)
    numeric_features = data.drop(['holiday', 'weekday', ], axis=1)
    scaler = StandardScaler()
    numeric_data = pd.DataFrame(scaler.fit_transform(numeric_features),
                                columns=numeric_features.columns,
                                index=numeric_features.index)
    # return pd.concat([data[['weekday', 'holiday']], numeric_data], axis=1)
    return pd.concat([data[['holiday', 'weekday']], numeric_data], axis=1)
