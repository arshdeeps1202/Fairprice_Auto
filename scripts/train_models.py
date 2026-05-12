import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_processed_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Processed file not found: {path}')
    return pd.read_csv(path)


def make_feature_matrix(df):
    if 'price' not in df.columns:
        raise KeyError('Processed data must include a price column.')

    y = df['price'].astype(float)
    X = df.drop(columns=['price'])

    # Drop derived or raw columns that should not be used directly when redundant
    drop_columns = ['year', 'title_status']
    for col in drop_columns:
        if col in X.columns:
            X = X.drop(columns=[col])

    categorical_columns = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, numeric_columns, categorical_columns


def build_transformer(numeric_columns, categorical_columns):
    transformers = []
    if numeric_columns:
        transformers.append(('num', StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
        )

    if not transformers:
        raise ValueError('No numeric or categorical features were found for the transformer.')

    return ColumnTransformer(transformers=transformers)


def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'drop' or name == 'remainder':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            named = transformer.get_feature_names_out(columns)
        else:
            named = [f'{name}__{col}' for col in columns]
        feature_names.extend(named)
    return feature_names


def save_eda_figures(df):
    ensure_dir('figures')

    plt.figure(figsize=(8, 5))
    sns.histplot(df['price'], bins=30, kde=True, color='steelblue')
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'price_distribution.png'))
    plt.close()

    if 'odometer' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='odometer', y='price', data=df, alpha=0.7)
        plt.title('Price vs. Mileage')
        plt.xlabel('Odometer')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'price_vs_mileage.png'))
        plt.close()

    if 'manufacturer' in df.columns:
        manuf = df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        plt.barh(manuf.index, manuf.values, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(manuf))))
        plt.title('Average Price by Manufacturer')
        plt.xlabel('Average Price')
        plt.ylabel('Manufacturer')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'average_price_by_manufacturer.png'))
        plt.close()

    if 'vehicle_age' in df.columns:
        age = df.groupby('vehicle_age')['price'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='vehicle_age', y='price', data=age, marker='o')
        plt.title('Average Price by Vehicle Age')
        plt.xlabel('Vehicle Age')
        plt.ylabel('Average Price')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'average_price_by_vehicle_age.png'))
        plt.close()


def train_models():
    ensure_dir('results')
    ensure_dir('figures')

    df = load_processed_data(os.path.join('data', 'processed', 'used_cars_processed.csv'))
    save_eda_figures(df)

    X, y, numeric_cols, categorical_cols = make_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    transformer = build_transformer(numeric_cols, categorical_cols)
    models = {
        'DummyRegressor': DummyRegressor(strategy='mean'),
        'LinearRegression': LinearRegression(),
        'RidgeRegression': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_iter_no_change=5, validation_fraction=0.1)
    }

    try:
        from xgboost import XGBRegressor
    except Exception:
        XGBRegressor = None

    if XGBRegressor is not None:
        models['XGBoost'] = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)

    metrics = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[('preprocessor', transformer), ('model', model)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        metrics.append({'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2})
        fitted_pipelines[name] = pipeline

    metrics_df = pd.DataFrame(metrics).sort_values('rmse')
    metrics_df.to_csv(os.path.join('results', 'model_metrics.csv'), index=False)

    best_model_name = metrics_df.iloc[0]['model']
    best_pipeline = fitted_pipelines[best_model_name]
    best_predictions = best_pipeline.predict(X_test)

    results_df = pd.DataFrame({
        'actual_price': y_test,
        'predicted_price': best_predictions
    }).reset_index(drop=True)
    results_df['price_difference'] = results_df['predicted_price'] - results_df['actual_price']
    results_df['percent_difference'] = results_df['price_difference'] / results_df['actual_price']
    results_df.to_csv(os.path.join('results', 'predictions.csv'), index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='rmse', y='model', data=metrics_df, color='steelblue')
    plt.title('Model Comparison (RMSE)')
    plt.xlabel('RMSE')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'model_comparison.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=results_df['actual_price'], y=results_df['predicted_price'], alpha=0.8)
    plt.plot([results_df['actual_price'].min(), results_df['actual_price'].max()],
             [results_df['actual_price'].min(), results_df['actual_price'].max()],
             color='red', linestyle='--')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'actual_vs_predicted.png'))
    plt.close()

    residuals = results_df['predicted_price'] - results_df['actual_price']
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=25, kde=True, color='purple')
    plt.title('Residual Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'residual_distribution.png'))
    plt.close()

    best_model = best_pipeline.named_steps['model']
    if hasattr(best_model, 'feature_importances_'):
        feature_names = get_feature_names(transformer)
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance, palette='magma')
        plt.title(f'Feature Importances for {best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'feature_importance.png'))
        plt.close()
    else:
        print('Best model does not support feature importances. Skipping that figure.')

    print(f'Best model: {best_model_name}')
    print(f'Model metrics saved to results/model_metrics.csv')
    print(f'Predictions saved to results/predictions.csv')


if __name__ == '__main__':
    train_models()
