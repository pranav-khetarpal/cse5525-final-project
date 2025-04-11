import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="VAR training loop")

    parser.add_argument("--train_file", type=str, default="tweet/VAR_stockemo_train.csv")
    parser.add_argument("--val_file", type=str, default="tweet/VAR_stockemo_val.csv")
    parser.add_argument("--test_file", type=str, default="tweet/VAR_stockemo_test.csv")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="How should we name this experiment?")

    args = parser.parse_args()
    return args

def load_var_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path, parse_dates=["date"], index_col="date")
    val_df = pd.read_csv(val_path, parse_dates=["date"], index_col="date")
    test_df = pd.read_csv(test_path, parse_dates=["date"], index_col="date")

    # Normalization of data since ohlcv will range heavily quantifiably so just normalize Z-score validation
    combined_full_train_data = pd.concat([train_df, val_df])
    mean, std = combined_full_train_data.mean(), combined_full_train_data.std()

    train_df = (train_df - mean) / std
    val_df = (val_df - mean) / std
    test_df = (test_df - mean) / std

    return train_df, val_df, test_df, combined_full_train_data

def train_VAR_model(combined_full_train_data, max_lag=10):
    """
    Trains the VAR model on combined train+val data and selects the best lag using BIC.
    Returns the full training data and selected lag.
    """
    
    # Our sentiment scores are being considered automatically during training of VAR
    model = VAR(combined_full_train_data)

    # Range [1-max] -> max is inclusive for lag
    lag_order_results = model.select_order(maxlags=max_lag)
    # Selected lag returns most optimal lag between the range [1-max] based on the bic criteria

    # BIC is most preferred in financial setting due to its conservativeness (frugal model penalizing heavily for complexity) 
    selected_lag = lag_order_results.selected_orders["bic"]

    print(f"Selected lag order (BIC): {selected_lag}")

    return selected_lag

def eval_VAR_model(full_train_df, test_df, selected_lag):
    """
    Performs rolling forecast on the test set using the trained VAR model.
    Prints and returns MSE and directional accuracy.
    """
    predictions, actuals, direction_correct = [], [], []
    history = full_train_df.copy()

    for i in range(len(test_df)):
        model = VAR(history)
        fitted = model.fit(selected_lag)
        
        # Forecasted prices for the next day/time step
        forecast = fitted.forecast(history.values[-selected_lag:], steps=1)[0]

        # Predicting 'close'
        predictions.append(forecast[0])
        actual = test_df.iloc[i]["close"]
        actuals.append(actual)

        prev_close = history.iloc[-1]["close"]
        direction_pred = np.sign(forecast[0] - prev_close)
        direction_actual = np.sign(actual - prev_close)
        direction_correct.append(direction_pred == direction_actual)

        # Uppdate the history with the actual test point
        history = pd.concat([history, test_df.iloc[[i]]])

    mse = mean_squared_error(actuals, predictions)
    directional_accuracy = np.mean(direction_correct)

    print(f"\nMSE: {mse:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.4f}")

    return mse, directional_accuracy


def main():
    args = get_args()
    train_df, val_df, test_df = load_var_data(args.train_file, args.val_file, args.test_file)
    full_train, selected_lag = train_VAR_model(train_df, val_df)
    eval_VAR_model(full_train, test_df, selected_lag)


if __name__ == "__main__":
    main()