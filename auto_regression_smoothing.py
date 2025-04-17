import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import argparse
import warnings

OPTIMAL_WINDOW_SIZE = 19
OPTIMAL_LAGS = 8

def get_args():
    parser = argparse.ArgumentParser(description="AutoRegression training loop")

    parser.add_argument("--train_file", type=str, default="price/merged_closing_prices_train.csv")
    parser.add_argument("--val_file", type=str, default="price/merged_closing_prices_val.csv")
    parser.add_argument("--test_file", type=str, default="price/merged_closing_prices_test.csv")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Name for this experiment")
    parser.add_argument("--lags", type=int, default=OPTIMAL_LAGS, help="Number of lags for AutoRegression")
    parser.add_argument("--window", type=int, default=OPTIMAL_WINDOW_SIZE, help="Number of days for Window Size")

    return parser.parse_args()

def load_data(train_path, val_path, test_path, window):
    train_df = pd.read_csv(train_path, parse_dates=["date"], index_col="date")
    val_df = pd.read_csv(val_path, parse_dates=["date"], index_col="date")
    test_df = pd.read_csv(test_path, parse_dates=["date"], index_col="date")

    # Drop columns with any NaNs
    all_data = pd.concat([train_df, val_df, test_df])
    clean_columns = all_data.dropna(axis=1).columns
    dropped_columns = set(all_data.columns) - set(clean_columns)
    if dropped_columns:
        print(f"[WARNING] Dropped columns due to NaNs: {sorted(dropped_columns)}")
    else:
        print("[INFO] No columns dropped; all columns are complete.")

    train_df = train_df[clean_columns]
    val_df = val_df[clean_columns]
    test_df = test_df[clean_columns]

    # Normalize
    mean, std = train_df.mean(), train_df.std()
    train_df = (train_df - mean) / std
    val_df = (val_df - mean) / std
    test_df = (test_df - mean) / std

    # Apply Simple Moving Average Smoothing (window=5 by default)
    train_df = train_df.rolling(window=window).mean()
    val_df = val_df.rolling(window=window).mean()
    test_df = test_df.rolling(window=window).mean()

    return train_df, val_df, test_df

def evaluate_autoreg(train_df, val_df, lags):
    warnings.filterwarnings("ignore")

    results = {}
    for col in train_df.columns:
        print(f"Training AutoReg for {col}...")

        train_series = train_df[col].dropna()
        val_series = val_df[col].dropna()

        predictions = []
        actuals = []
        direction_correct = []

        history = train_series.copy()

        for i in range(len(val_series)):
            try:
                model = AutoReg(history, lags=lags, old_names=False).fit()
                pred = model.predict(start=len(history), end=len(history)).iloc[0]

            except Exception as e:
                print(f"[ERROR] Skipping {col} at step {i} due to: {e}")
                break

            actual = val_series.iloc[i]
            prev = history.iloc[-1]

            predictions.append(pred)
            actuals.append(actual)
            direction_correct.append(np.sign(pred - prev) == np.sign(actual - prev))

            history = pd.concat([history, pd.Series([actual], index=[val_series.index[i]])])

        mse = mean_squared_error(actuals, predictions)
        directional_accuracy = np.mean(direction_correct)

        results[col] = {"mse": mse, "directional_accuracy": directional_accuracy}

    return results

def test_autoreg(train_val_df, test_df, lags):
    warnings.filterwarnings("ignore")

    results = {}
    for col in train_val_df.columns:
        print(f"Training AutoReg for {col}...")

        train_val_series = train_val_df[col].dropna()
        test_series = test_df[col].dropna()

        predictions = []
        actuals = []
        direction_correct = []

        history = train_val_series.copy()

        for i in range(len(test_series)):
            try:
                model = AutoReg(history, lags=lags, old_names=False).fit()
                pred = model.predict(start=len(history), end=len(history)).iloc[0]

            except Exception as e:
                print(f"[ERROR] Skipping {col} at step {i} due to: {e}")
                break

            actual = test_series.iloc[i]
            prev = history.iloc[-1]

            predictions.append(pred)
            actuals.append(actual)
            direction_correct.append(np.sign(pred - prev) == np.sign(actual - prev))

            history = pd.concat([history, pd.Series([actual], index=[test_series.index[i]])])

        mse = mean_squared_error(actuals, predictions)
        directional_accuracy = np.mean(direction_correct)

        results[col] = {"mse": mse, "directional_accuracy": directional_accuracy}

    return results

def main():
    args = get_args()
    train_df, val_df, test_df = load_data(args.train_file, args.val_file, args.test_file, args.window)
    validation_results = evaluate_autoreg(train_df, val_df, args.lags)

    total_directional_accuracy = 0
    total_mse = 0
    total_stocks_counter = 0

    print("\nValidation Results:")
    for stock, metrics in validation_results.items():
        print(f"{stock}: MSE = {metrics['mse']:.4f}, Directional Accuracy = {metrics['directional_accuracy']:.4f}")
        total_directional_accuracy += metrics['directional_accuracy']
        total_mse += metrics['mse']
        total_stocks_counter += 1
    
    print(f"Lag of {args.lags}. Average MSE: {total_mse / total_stocks_counter}, Average Directional Accuracy: {total_directional_accuracy / total_stocks_counter}")

    train_val_df = pd.concat([train_df, val_df])
    test_results = test_autoreg(train_val_df, test_df, args.lags)

    test_total_directional_accuracy = 0
    test_total_mse = 0
    test_total_stocks_counter = 0

    print("\nTest Results:")
    for stock, metrics in test_results.items():
        print(f"{stock}: MSE = {metrics['mse']:.4f}, Directional Accuracy = {metrics['directional_accuracy']:.4f}")
        test_total_directional_accuracy += metrics['directional_accuracy']
        test_total_mse += metrics['mse']
        test_total_stocks_counter += 1

    print(f"Lag of {args.lags}. Average MSE: {test_total_mse / test_total_stocks_counter}, Average Directional Accuracy: {test_total_directional_accuracy / test_total_stocks_counter}")

if __name__ == "__main__":
    main()
    