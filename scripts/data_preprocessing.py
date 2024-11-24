# scripts/data_preprocessing.py
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    # Handle missing values
    print(f"Number of missing data \n{data.isnull().sum()}")
    # Drop columns with no valuable information
    if 'snow' in data.columns and data['snow'].nunique() == 1:
        data = data.drop(columns=['snow'])
    return data
