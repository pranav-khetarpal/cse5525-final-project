import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="VAR training loop")

    parser.add_argument(
        "--train_file", type=str, default="price/merged_closing_prices_train.csv"
    )
    parser.add_argument(
        "--val_file", type=str, default="price/merged_closing_prices_val.csv"
    )
    parser.add_argument(
        "--test_file", type=str, default="price/merged_closing_prices_test.csv"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="How should we name this experiment?",
    )

    args = parser.parse_args()
    return args


def load_var_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path, parse_dates=["Date"], index_col="Date")
    val_df = pd.read_csv(val_path, parse_dates=["Date"], index_col="Date")
    test_df = pd.read_csv(test_path, parse_dates=["Date"], index_col="Date")

    # Normalization of data since ohlcv will range heavily quantifiably so just normalize Z-score validation
    mean, std = train_df.mean(), train_df.std()

    train_df = (train_df - mean) / std
    val_df = (val_df - mean) / std
    test_df = (test_df - mean) / std

    return train_df, val_df, test_df


def train_VAR_model(train_df, max_lag=10):
    """
    Trains the VAR model on combined train+val data and selects the best lag using BIC.
    Returns the full training data and selected lag.
    """

    print(train_df.head(5))

    # Our sentiment scores are being considered automatically during training of VAR
    model = VAR(train_df)

    # Range [1-max] -> max is inclusive for lag
    lag_order_results = model.select_order(maxlags=max_lag)
    # Selected lag returns most optimal lag between the range [1-max] based on the bic criteria

    # BIC is most preferred in financial setting due to its conservativeness (frugal model penalizing heavily for complexity)
    selected_lag = lag_order_results.selected_orders["bic"]

    print(f"Selected lag order (BIC): {selected_lag}")

    return selected_lag


def eval_VAR_model_on_val(full_train_df, val_df, selected_lag):
    """
    Performs rolling forecast on the test set using the trained VAR model.
    Prints and returns MSE and directional accuracy.
    """
    predictions, actuals, direction_correct = [], [], []
    history = full_train_df.copy()

    for i in range(len(val_df)):
        model = VAR(history)
        fitted = model.fit(selected_lag)

        # Forecasted prices for the next day/time step
        forecast = fitted.forecast(history.values[-selected_lag:], steps=1)[0]

        # Predicting 'close'
        predictions.append(forecast[0])
        actual = val_df.iloc[i]["close"]
        actuals.append(actual)

        prev_close = history.iloc[-1]["close"]
        direction_pred = np.sign(forecast[0] - prev_close)
        direction_actual = np.sign(actual - prev_close)
        direction_correct.append(direction_pred == direction_actual)

        # Uppdate the history with the actual test point
        history = pd.concat([history, val_df.iloc[[i]]])

    mse = mean_squared_error(actuals, predictions)
    directional_accuracy = np.mean(direction_correct)

    print(f"\n Validation MSE: {mse:.4f}")
    print(f"Validation Directional Accuracy: {directional_accuracy:.4f}")

    return mse, directional_accuracy, history


# def eval_VAR_model_on_test(full_train_df, test_df, selected_lag):
#     """
#     Performs rolling forecast on the test set using the trained VAR model.
#     Prints and returns MSE and directional accuracy.
#     """
#     predictions, actuals, direction_correct = [], [], []
#     history = full_train_df.copy()

#     for i in range(len(test_df)):
#         model = VAR(history)
#         fitted = model.fit(selected_lag)

#         # Forecasted prices for the next day/time step
#         forecast = fitted.forecast(history.values[-selected_lag:], steps=1)[0]

#         # Predicting 'close'
#         predictions.append(forecast[0])
#         actual = test_df.iloc[i]["close"]
#         actuals.append(actual)

#         prev_close = history.iloc[-1]["close"]
#         direction_pred = np.sign(forecast[0] - prev_close)
#         direction_actual = np.sign(actual - prev_close)
#         direction_correct.append(direction_pred == direction_actual)

#         # Uppdate the history with the actual test point
#         history = pd.concat([history, test_df.iloc[[i]]])

#     mse = mean_squared_error(actuals, predictions)
#     directional_accuracy = np.mean(direction_correct)

#     print(f"\nMSE: {mse:.4f}")
#     print(f"Directional Accuracy: {directional_accuracy:.4f}")

#     return mse, directional_accuracy, history


def main():
    args = get_args()
    # Train for training, val for mini test, test for final big test
    train_df, val_df, test_df = load_var_data(
        args.train_file, args.val_file, args.test_file
    )
    selected_lag = train_VAR_model(train_df, 1)

    val_mse, val_directional_accuracy, val_history = eval_VAR_model_on_val(
        train_df, val_df, selected_lag
    )
    # test_mse, test_directional_accuracy, test_history = eval_VAR_model_on_test(val_history, test_df, selected_lag)


if __name__ == "__main__":
    main()
