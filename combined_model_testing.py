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
        default=0.05,
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
        default=0.05,
        help="Step size for confidence threshold grid search",
    )

    # Add new arguments for boost_confidence
    parser.add_argument(
        "--boost_confidence_min",
        type=float,
        default=0.1,
        help="Minimum boost confidence to try",
    )

    parser.add_argument(
        "--boost_confidence_max",
        type=float,
        default=0.5,
        help="Maximum boost confidence to try",
    )

    parser.add_argument(
        "--boost_confidence_step",
        type=float,
        default=0.05,
        help="Step size for boost confidence grid search",
    )

    return parser.parse_args()


def evaluate_params(
    model_name, base_confidence, confidence_threshold, boost_confidence
):
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

            # Make prediction with current parameters, including boost_confidence
            prediction = prediction_for_stock_and_day(
                auto_regression_direction,
                average_sentiment,
                base_confidence,
                confidence_threshold,
                boost_confidence,
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

    boost_confidence_range = np.arange(
        args.boost_confidence_min,
        args.boost_confidence_max + args.boost_confidence_step / 2,
        args.boost_confidence_step,
    )

    # Round to avoid floating point issues
    base_confidence_range = np.round(base_confidence_range, 2)
    confidence_threshold_range = np.round(confidence_threshold_range, 2)
    boost_confidence_range = np.round(boost_confidence_range, 2)

    print(
        f"Parameter grid search with {len(base_confidence_range) * len(confidence_threshold_range) * len(boost_confidence_range)} combinations"
    )

    # Store results
    results = []
    best_accuracy = 0
    best_params = None

    # Progress counter
    total_combos = len(base_confidence_range) * len(confidence_threshold_range) * len(boost_confidence_range)
    current_combo = 0

    # Grid search over all three parameters
    for base_confidence in base_confidence_range:
        for confidence_threshold in confidence_threshold_range:
            # Skip invalid combinations where threshold > base_confidence
            if confidence_threshold >= base_confidence:
                current_combo += len(boost_confidence_range)
                continue

            for boost_confidence in boost_confidence_range:
                current_combo += 1
                print(
                    f"Evaluating combination {current_combo}/{total_combos}: "
                    f"base_confidence={base_confidence}, confidence_threshold={confidence_threshold}, "
                    f"boost_confidence={boost_confidence}"
                )

                # Evaluate this parameter combination
                accuracy, correct, total = evaluate_params(
                    args.model_name, base_confidence, confidence_threshold, boost_confidence
                )

                # Store result
                results.append(
                    {
                        "base_confidence": base_confidence,
                        "confidence_threshold": confidence_threshold,
                        "boost_confidence": boost_confidence,
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
                        "boost_confidence": boost_confidence,
                    }
                    print(f"New best: accuracy={accuracy:.4f}, params={best_params}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, best_params, best_accuracy


def visualize_top_results(results_df, model_name):
    """
    Create publication-quality visualization of parameter search results
    """
    if len(results_df) == 0:
        print("No results to visualize")
        return

    # Sort by accuracy and get top results
    top_results = results_df.sort_values("accuracy", ascending=False).head(20)
    best_result = top_results.iloc[0]

    # Set publication-quality styling
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))

    # 1. Main 3D scatter plot
    ax1 = fig.add_subplot(221, projection="3d")

    # Create custom colormap with clear distinction
    cmap = plt.cm.viridis
    norm = plt.Normalize(top_results["accuracy"].min(), top_results["accuracy"].max())

    scatter = ax1.scatter(
        top_results["base_confidence"],
        top_results["confidence_threshold"],
        top_results["boost_confidence"],
        c=top_results["accuracy"],
        cmap=cmap,
        s=150,
        alpha=0.8,
        edgecolors="w",
        linewidth=0.5,
    )

    # Highlight the best result
    ax1.scatter(
        [best_result["base_confidence"]],
        [best_result["confidence_threshold"]],
        [best_result["boost_confidence"]],
        c="red",
        s=250,
        marker="*",
        label="Best Parameters",
    )

    # Improve axes
    ax1.set_xlabel("Base Confidence", fontweight="bold")
    ax1.set_ylabel("Confidence Threshold", fontweight="bold")
    ax1.set_zlabel("Boost Confidence", fontweight="bold")

    # Add gridlines
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Add colorbar with meaningful ticks
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
    cbar.set_label("Prediction Accuracy", fontweight="bold")

    # Add legend
    ax1.legend(loc="upper left")

    # 2. Create 2D plot for Base Confidence vs Confidence Threshold
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(
        top_results["base_confidence"],
        top_results["confidence_threshold"],
        c=top_results["accuracy"],
        cmap=cmap,
        s=150,
        alpha=0.8,
        edgecolors="w",
    )

    # Highlight best
    ax2.scatter(
        best_result["base_confidence"],
        best_result["confidence_threshold"],
        c="red",
        s=250,
        marker="*",
    )

    ax2.set_xlabel("Base Confidence", fontweight="bold")
    ax2.set_ylabel("Confidence Threshold", fontweight="bold")
    ax2.set_title("Base Confidence vs Confidence Threshold", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Create 2D plot for Base Confidence vs Boost Confidence
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(
        top_results["base_confidence"],
        top_results["boost_confidence"],
        c=top_results["accuracy"],
        cmap=cmap,
        s=150,
        alpha=0.8,
        edgecolors="w",
    )

    # Highlight best
    ax3.scatter(
        best_result["base_confidence"],
        best_result["boost_confidence"],
        c="red",
        s=250,
        marker="*",
    )

    ax3.set_xlabel("Base Confidence", fontweight="bold")
    ax3.set_ylabel("Boost Confidence", fontweight="bold")
    ax3.set_title("Base Confidence vs Boost Confidence", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Create table of top 5 results
    ax4 = fig.add_subplot(224)
    ax4.axis("off")

    top5 = top_results.head(5).reset_index(drop=True)
    top5_display = top5[
        ["base_confidence", "confidence_threshold", "boost_confidence", "accuracy"]
    ]
    top5_display = top5_display.round(4)

    # Create table with top 5 results
    table_data = [
        ["Rank", "Base Conf", "Thresh", "Boost", "Accuracy"],
    ]

    for i, row in top5_display.iterrows():
        table_data.append(
            [
                f"{i+1}",
                f"{row['base_confidence']:.2f}",
                f"{row['confidence_threshold']:.2f}",
                f"{row['boost_confidence']:.2f}",
                f"{row['accuracy']:.4f}",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.1, 0.2, 0.2, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Color the header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(5)):
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight the best result row
    for j, cell in enumerate(table._cells[(1, j)] for j in range(5)):
        cell.set_facecolor("#E2EFDA")

    ax4.set_title("Top 5 Parameter Combinations", fontweight="bold")

    # Set main title
    plt.suptitle(
        f"Parameter Optimization Results for {model_name} Model",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Add description
    description = (
        f"Grid search over parameter space to optimize prediction accuracy.\n"
        f"Best parameters: Base Conf={best_result['base_confidence']:.2f}, "
        f"Threshold={best_result['confidence_threshold']:.2f}, "
        f"Boost={best_result['boost_confidence']:.2f}\n"
        f"Best accuracy: {best_result['accuracy']:.4f}"
    )

    fig.text(0.5, 0.02, description, ha="center", fontsize=12, fontstyle="italic")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save in high resolution
    plt.savefig(
        f"parameter_optimization_{model_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"parameter_optimization_{model_name}.pdf", format="pdf", bbox_inches="tight"
    )

    print(
        f"Publication-quality visualization saved as parameter_optimization_{model_name}.png and .pdf"
    )

    return fig


def main():
    args = get_args()

    print(f"Starting parameter grid search for {args.model_name} model")

    # Perform grid search
    results_df, best_params, best_accuracy = grid_search(args)

    # Save results to CSV
    results_file = f"combined_model_data/parameter_search_results_{args.model_name}2.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Print best parameters
    print("\n=== BEST PARAMETERS ===")
    print(f"Model: {args.model_name}")
    print(f"Base Confidence: {best_params['base_confidence']}")
    print(f"Confidence Threshold: {best_params['confidence_threshold']}")
    print(f"Boost Confidence: {best_params['boost_confidence']}")
    print(f"Accuracy: {best_accuracy:.4f}")

    # Save best parameters to file
    best_params_file = f"combined_model_data/best_parameters2_{args.model_name}.txt"
    with open(best_params_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Base Confidence: {best_params['base_confidence']}\n")
        f.write(f"Confidence Threshold: {best_params['confidence_threshold']}\n")
        f.write(f"Boost Confidence: {best_params['boost_confidence']}\n")
        f.write(f"Accuracy: {best_accuracy:.4f}\n")
    print(f"Best parameters saved to {best_params_file}")

    # Visualize top results
    try:
        visualize_top_results(results_df, args.model_name)
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")


if __name__ == "__main__":
    args = get_args()
    main()
