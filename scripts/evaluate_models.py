import os
import pandas as pd


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    return pd.read_csv(path)


def summarize(df, columns):
    return df[columns].describe().transpose()


def run_evaluation():
    ensure_dir = os.makedirs
    ensure_dir('results', exist_ok=True)

    metrics_path = os.path.join('results', 'model_metrics.csv')
    predictions_path = os.path.join('results', 'predictions.csv')
    summary_path = os.path.join('results', 'final_summary.txt')

    metrics_df = load_csv(metrics_path)
    predictions_df = load_csv(predictions_path)

    print('Model Comparison Metrics:')
    print(metrics_df.to_string(index=False))
    print('\nPrediction Summary:')

    summary_stats = summarize(predictions_df, ['actual_price', 'predicted_price', 'price_difference', 'percent_difference'])
    print(summary_stats.to_string())

    if 'fairness_label' in predictions_df.columns:
        label_counts = predictions_df['fairness_label'].value_counts()
        print('\nFairness Label Counts:')
        print(label_counts.to_string())
    else:
        label_counts = pd.Series(dtype=int)
        print('\nFairness labels not found in predictions.csv.')

    with open(summary_path, 'w') as file:
        file.write('Model Comparison Metrics\n')
        file.write(metrics_df.to_string(index=False))
        file.write('\n\nPrediction Summary Statistics\n')
        file.write(summary_stats.to_string())
        if not label_counts.empty:
            file.write('\n\nFairness Label Counts\n')
            file.write(label_counts.to_string())

    print(f'Final summary saved to {summary_path}')


if __name__ == '__main__':
    run_evaluation()
