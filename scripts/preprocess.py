import os
import datetime
import numpy as np
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def safe_column_names(df, cols):
    return [col for col in cols if col in df.columns]


def preprocess():
    raw_path = os.path.join('data', 'raw', 'used_cars_raw.csv')
    output_dir = os.path.join('data', 'processed')
    output_path = os.path.join(output_dir, 'used_cars_processed.csv')

    ensure_dir(output_dir)
    df = load_data(raw_path)

    df = df.drop_duplicates()

    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'].notna()]
    else:
        raise KeyError('The raw dataset must include a price column.')

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        df['year'] = np.nan

    if 'odometer' in df.columns:
        df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    else:
        df['odometer'] = np.nan

    df = df[df['price'].between(1000, 150000)]
    df = df[df['odometer'].between(0, 400000)]

    current_year = datetime.datetime.now().year
    df['vehicle_age'] = current_year - df['year']
    df = df[~((df['vehicle_age'] < 0) | (df['vehicle_age'] > 40))]

    df['mileage_per_year'] = df['odometer'] / df['vehicle_age'].clip(lower=1)

    if 'title_status' in df.columns:
        df['is_clean_title'] = df['title_status'].astype(str).str.lower().str.contains('clean').astype(int)
    else:
        df['is_clean_title'] = 0

    luxury_brands = {
        'bmw', 'mercedes-benz', 'audi', 'lexus', 'porsche', 'tesla',
        'cadillac', 'infiniti', 'acura', 'genesis', 'jaguar', 'land rover'
    }
    if 'manufacturer' in df.columns:
        df['is_luxury_brand'] = df['manufacturer'].astype(str).str.lower().isin(luxury_brands).astype(int)
    else:
        df['is_luxury_brand'] = 0

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown').astype(str)

    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'price']
    for col in numeric_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    df.to_csv(output_path, index=False)
    print(f'Processed data saved to {output_path}')


if __name__ == '__main__':
    preprocess()
