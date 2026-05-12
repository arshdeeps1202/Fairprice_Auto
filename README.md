# FairPrice Auto: Explainable Machine Learning for Used Car Price Fairness Detection

## Project Overview
This project uses a used car listings dataset to build a regression model that predicts fair market price. It then labels each listing as Underpriced, Fairly Priced, or Overpriced based on the difference between the actual listing price and the predicted price.

## Research Question
Can we predict a fair market price for a used car listing and identify listings that are significantly undervalued or overvalued?

## Dataset Description
The raw dataset is expected at `data/raw/used_cars_raw.csv`. It may include columns such as:
- `price`
- `year`
- `manufacturer`
- `model`
- `condition`
- `cylinders`
- `fuel`
- `odometer`
- `title_status`
- `transmission`
- `drive`
- `type`
- `paint_color`
- `state`

The preprocessing script removes duplicates, invalid prices and mileage, derives age and mileage features, and encodes useful indicators.

The fairness report file `results/fairness_classification_report.csv` contains counts and percentages of each price-risk label.

## Folder Structure
```
FairPrice-Auto/
│
├── data/
│   ├── raw/
│   │   └── used_cars_raw.csv
│   └── processed/
│       └── used_cars_processed.csv
│
├── scripts/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── fairness_labeling.py
│
├── notebooks/
│   └── eda_modeling.ipynb
│
├── figures/
│   ├── price_distribution.png
│   ├── price_vs_mileage.png
│   ├── average_price_by_manufacturer.png
│   ├── average_price_by_vehicle_age.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residual_distribution.png
│   ├── feature_importance.png
│   ├── fairness_distribution.png
│   └── fairness_confusion_matrix.png
│
├── results/
│   ├── model_metrics.csv
│   ├── fairness_classification_report.csv
│   └── predictions.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project
From the project root directory, run:

```bash
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/train_models.py
python scripts/fairness_labeling.py
python scripts/evaluate_models.py
```

## Preprocessing Explanation
The preprocessing pipeline:
- removes duplicates
- drops rows with missing prices
- converts `price`, `year`, and `odometer` to numeric values
- removes unrealistic prices and mileages
- computes `vehicle_age` and `mileage_per_year`
- derives binary flags for clean title and luxury brand
- fills missing categorical values with `Unknown`
- fills missing numeric values with the median

## Models
The training script compares multiple regressors:
- baseline mean predictor
- linear regression
- ridge regression
- random forest regression
- gradient boosting regression
- XGBoost regression (if installed)

## Evaluation Metrics
Model performance is measured by:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R2 Score

## Price Fairness Labels
Listings are labeled using percent difference:
- `Underpriced` when percent difference ≤ -10%
- `Fairly Priced` when percent difference is between -10% and 10%
- `Overpriced` when percent difference ≥ 10%

## Limitations and Future Work
Current limitations:
- model uses only structured listing fields
- price prediction may still miss market trends and local condition impacts
- future work could incorporate more features, dealer vs private listings, and geographic pricing trends
