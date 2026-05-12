import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def label_fairness(row):
    pct = row['percent_difference']
    if pct <= -0.10:
        return 'Underpriced'
    if pct >= 0.10:
        return 'Overpriced'
    return 'Fairly Priced'


def run_fairness_labeling():
    predictions_path = os.path.join('results', 'predictions.csv')
    if not os.path.exists(predictions_path):
        raise FileNotFoundError('Predictions file not found. Run train_models.py first.')

    df = pd.read_csv(predictions_path)
    if 'percent_difference' not in df.columns:
        df['percent_difference'] = (df['predicted_price'] - df['actual_price']) / df['actual_price']

    df['fairness_label'] = df.apply(label_fairness, axis=1)
    df.to_csv(predictions_path, index=False)

    ensure_dir('figures')
    ensure_dir('results')

    label_counts = df['fairness_label'].value_counts().reindex(['Underpriced', 'Fairly Priced', 'Overpriced']).fillna(0)
    summary_df = pd.DataFrame({
        'fairness_label': label_counts.index,
        'count': label_counts.values,
        'percentage': (label_counts.values / label_counts.values.sum()) * 100
    })
    report_path = os.path.join('results', 'fairness_classification_report.csv')
    summary_df.to_csv(report_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(label_counts.index, label_counts.values, color=['#8ecae6', '#219ebc', '#ffb703'])
    plt.title('Fairness Label Distribution')
    plt.xlabel('Fairness Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'fairness_distribution.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.heatmap(label_counts.to_frame().T, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Fairness Label Summary')
    plt.xlabel('Fairness Label')
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'fairness_confusion_matrix.png'))
    plt.close()

    print(f'Fairness labels added and saved to {predictions_path}')
    print('Fairness distribution chart saved to figures/fairness_distribution.png')
    print(f'Fairness summary report saved to {report_path}')


if __name__ == '__main__':
    run_fairness_labeling()
