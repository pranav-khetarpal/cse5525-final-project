import pandas as pd
import argparse

BERT_TEST_SENTIMENT_PATH = "combined_model_data/BERT_average_sentiment_per_stock.csv"
FINANCIALBERT_TEST_SENTIMENT_PATH = "combined_model_data/FinancialBERT_average_sentiment_per_stock.csv"
AUTOREGRESSION_TEST_DIRECTION_PATH = "combined_model_data/auto_regression_test_predictions.csv"
ACTUAL_TEST_DIRECTION_PATH = "combined_model_data/actual_test_predictions.csv"

def get_args():
    parser = argparse.ArgumentParser(description="Combined model")

    parser.add_argument(
        "--model_name",
        type=str,
        default="BERT",
        choices=["BERT", "FinancialBERT"],
        help="What model to fine-tune",
    )

    parser.add_argument(
        "--base_confidence",
        type=float,
        default=0.6,
        help="What base confidence to use"
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.2,
        help="What confidence threshold to use"
    )

    return parser.parse_args()

def confidence_for_stock_and_day(auto_regression_direction, average_sentiment, base_confidence, confidence_threshold):
    """
    """

    # Convert to binary sentiment (will likely only be 0 or 1 if available)
    bert_direction = 1 if average_sentiment > 0.5 else 0

    # If both agree, boost confidence
    if auto_regression_direction == bert_direction:
        final_confidence = base_confidence + (0.2 * abs(average_sentiment - 0.5) * 2)
    else:
        # If disagree, reduce confidence based on sentiment strength
        final_confidence = base_confidence - (0.2 * abs(average_sentiment - 0.5) * 2)

    # Final prediction remains the regression direction, but with adjusted confidence
    final_prediction = auto_regression_direction
    confidence = final_confidence

    if confidence < confidence_threshold:
        if final_prediction == 1:
            final_prediction = 0
        else:
            final_prediction = 1

    return final_prediction

def main():
    """
    """
    args = get_args()
    
    if args.model_name == "BERT":
        sentiment_df = pd.read_csv(BERT_TEST_SENTIMENT_PATH)
    else:
        sentiment_df = pd.read_csv(FINANCIALBERT_TEST_SENTIMENT_PATH)

    auto_regression_direction_df = pd.read_csv(AUTOREGRESSION_TEST_DIRECTION_PATH)
    actual_direction_df = pd.read_csv(ACTUAL_TEST_DIRECTION_PATH)

    correct_predictions = 0
    total_predictions = 0

    stock_accurracies = {}

    for column in sentiment_df.columns:
        if column == "date":
            continue

        stock = column.spli('_')[-1] # get the stock name

        for auto_regression_column in auto_regression_direction_df.columns:
            if stock in auto_regression_column:
                
            average_sentiment = row[column]


if __name__ == "__main__":
    main()
    