import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from combined_model import (
    prediction_for_stock_and_day,
    BERT_TEST_SENTIMENT_PATH,
    FINANCIALBERT_TEST_SENTIMENT_PATH,
    AUTOREGRESSION_TEST_DIRECTION_PATH,
    ACTUAL_TEST_DIRECTION_PATH,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Find optimal parameters for combined model"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BERT",
        choices=["BERT", "FinancialBERT"],
        help="What sentiment model to use",
    )

    parser.add_argument(
        "--base_confidence_min",
        type=float,
        default=0.1,
        help="Minimum base confidence to try",
    )

    parser.add_argument(
        "--base_confidence_max",
        type=float,
        default=0.9,
        help="Maximum base confidence to try",
    )

    parser.add_argument(
        "--base_confidence_step",
        type=float,
        default=0.01,
        help="Step size for base confidence grid search",
    )

    parser.add_argument(
        "--confidence_threshold_min",
        type=float,
        default=0.1,
        help="Minimum confidence threshold to try",
    )

    parser.add_argument(
        "--confidence_threshold_max",
        type=float,
        default=0.5,
        help="Maximum confidence threshold to try",
    )

    parser.add_argument(
        "--confidence_threshold_step",
        type=float,
        default=0.01,
        help="Step size for confidence threshold grid search",
    )

    return parser.parse_args()


def evaluate_params(model_name, base_confidence, confidence_threshold):
    """
    Evaluates a specific parameter combination and returns the accuracy
    """
    # Load sentiment data
    if model_name == "BERT":
        sentiment_df = pd.read_csv(BERT_TEST_SENTIMENT_PATH)
    else:
        sentiment_df = pd.read_csv(FINANCIALBERT_TEST_SENTIMENT_PATH)

    # Load auto-regression predictions and actual directions
    auto_regression_df = pd.read_csv(AUTOREGRESSION_TEST_DIRECTION_PATH)
    actual_df = pd.read_csv(ACTUAL_TEST_DIRECTION_PATH)

    # Set date as index
    auto_regression_df.set_index("date", inplace=True)
    sentiment_df.set_index("date", inplace=True)
    actual_df.set_index("date", inplace=True)

    correct_predictions = 0
    total_predictions = 0

    # Process each stock
    for sentiment_column in sentiment_df.columns:
        if sentiment_column == "date":
            continue

        stock = sentiment_column.split("_")[-1]  # get the stock name

        # Find corresponding columns in other dataframes
        auto_regression_column = None
        for col in auto_regression_df.columns:
            if stock in col:
                auto_regression_column = col
                break

        if auto_regression_column is None:
            continue

        actual_column = None
        for col in actual_df.columns:
            if stock in col:
                actual_column = col
                break

        if actual_column is None:
            continue

        # Get common dates
        common_dates = auto_regression_df.index.intersection(
            sentiment_df.index
        ).intersection(actual_df.index)

        # Process each date
        for date in common_dates:
            if (
                pd.isna(auto_regression_df.loc[date, auto_regression_column])
                or pd.isna(sentiment_df.loc[date, sentiment_column])
                or pd.isna(actual_df.loc[date, actual_column])
            ):
                continue

            # Get values for this stock and day
            auto_regression_direction = int(
                auto_regression_df.loc[date, auto_regression_column]
            )
            average_sentiment = float(sentiment_df.loc[date, sentiment_column])
            actual_direction = int(actual_df.loc[date, actual_column])

            # Make prediction with current parameters
            prediction = prediction_for_stock_and_day(
                auto_regression_direction,
                average_sentiment,
                base_confidence,
                confidence_threshold,
            )

            # Check if prediction is correct
            if prediction == actual_direction:
                correct_predictions += 1

            total_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return accuracy, correct_predictions, total_predictions


def grid_search(args):
    """
    Perform grid search over parameter space to find optimal parameters
    """
    # Create parameter ranges
    base_confidence_range = np.arange(
        args.base_confidence_min,
        args.base_confidence_max + args.base_confidence_step / 2,
        args.base_confidence_step,
    )

    confidence_threshold_range = np.arange(
        args.confidence_threshold_min,
        args.confidence_threshold_max + args.confidence_threshold_step / 2,
        args.confidence_threshold_step,
    )

    # Round to avoid floating point issues
    base_confidence_range = np.round(base_confidence_range, 2)
    confidence_threshold_range = np.round(confidence_threshold_range, 2)

    print(
        f"Parameter grid search with {len(base_confidence_range) * len(confidence_threshold_range)} combinations"
    )

    # Store results
    results = []
    best_accuracy = 0
    best_params = None

    # Progress counter
    total_combos = len(base_confidence_range) * len(confidence_threshold_range)
    current_combo = 0

    # Grid search
    for base_confidence in base_confidence_range:
        for confidence_threshold in confidence_threshold_range:
            # Skip invalid combinations where threshold > base_confidence
            if confidence_threshold >= base_confidence:
                current_combo += 1
                continue

            current_combo += 1
            print(
                f"Evaluating combination {current_combo}/{total_combos}: "
                f"base_confidence={base_confidence}, confidence_threshold={confidence_threshold}"
            )

            # Evaluate this parameter combination
            accuracy, correct, total = evaluate_params(
                args.model_name, base_confidence, confidence_threshold
            )

            # Store result
            results.append(
                {
                    "base_confidence": base_confidence,
                    "confidence_threshold": confidence_threshold,
                    "accuracy": accuracy,
                    "correct_predictions": correct,
                    "total_predictions": total,
                }
            )

            # Update best parameters if needed
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "base_confidence": base_confidence,
                    "confidence_threshold": confidence_threshold,
                }
                print(f"New best: accuracy={accuracy:.4f}, params={best_params}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, best_params, best_accuracy


def visualize_results(results_df, model_name):
    """
    Create visualization of grid search results
    """
    if len(results_df) == 0:
        print("No results to visualize")
        return

    # Create pivot table for heatmap
    pivot = results_df.pivot_table(
        index="base_confidence", columns="confidence_threshold", values="accuracy"
    )

    # Create heatmap
    plt.figure(figsize=(10, 8))
    heatmap = plt.pcolor(pivot)
    plt.colorbar(heatmap, label="Accuracy")

    # Add labels
    plt.title(f"Parameter Grid Search Results ({model_name})")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Base Confidence")

    # Set tick labels
    plt.xticks(np.arange(0.5, len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(0.5, len(pivot.index)), pivot.index)

    # Save figure
    plt.savefig(f"parameter_grid_search_{model_name}.png")
    print(f"Visualization saved as parameter_grid_search_{model_name}.png")


def main():
    args = get_args()

    print(f"Starting parameter grid search for {args.model_name} model")

    # Perform grid search
    results_df, best_params, best_accuracy = grid_search(args)

    # Save results to CSV
    results_file = f"combined_model_data/parameter_search_results_{args.model_name}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Print best parameters
    print("\n=== BEST PARAMETERS ===")
    print(f"Model: {args.model_name}")
    print(f"Base Confidence: {best_params['base_confidence']}")
    print(f"Confidence Threshold: {best_params['confidence_threshold']}")
    print(f"Accuracy: {best_accuracy:.4f}")

    # Save best parameters to file
    best_params_file = f"combined_model_data/best_parameters_{args.model_name}.txt"
    with open(best_params_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Base Confidence: {best_params['base_confidence']}\n")
        f.write(f"Confidence Threshold: {best_params['confidence_threshold']}\n")
        f.write(f"Accuracy: {best_accuracy:.4f}\n")
    print(f"Best parameters saved to {best_params_file}")

    # Visualize results
    try:
        visualize_results(results_df, args.model_name)
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")


if __name__ == "__main__":
    args = get_args()
    main()
