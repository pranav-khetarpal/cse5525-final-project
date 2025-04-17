import pandas as pd
import argparse

BERT_TEST_SENTIMENT_PATH = "combined_model_data/BERT_average_sentiment_per_stock.csv"
FINANCIALBERT_TEST_SENTIMENT_PATH = "combined_model_data/FinancialBERT_average_sentiment_per_stock.csv"
AUTOREGRESSION_TEST_DIRECTION_PATH = "combined_model_data/auto_regression_test_predictions.csv"
ACTUAL_TEST_DIRECTION_PATH = "combined_model_data/actual_test_prediction.csv"

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

def prediction_for_stock_and_day(auto_regression_direction, average_sentiment, base_confidence, confidence_threshold):
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
    
    # print(f"regression prediction: {final_prediction}")
    # print(f"sentiment score {average_sentiment}")

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
    stock_accuracies = {}

    auto_regression_direction_df.set_index('date', inplace=True)
    sentiment_df.set_index('date', inplace=True)
    actual_direction_df.set_index('date', inplace=True)

    for sentiment_column in sentiment_df.columns:
        if sentiment_column == "date":
            continue

        stock = sentiment_column.split('_')[-1] # get the stock name

        auto_regression_column = None
        for auto_regression_col in auto_regression_direction_df.columns:
            if stock in auto_regression_col:
                auto_regression_column = auto_regression_col
                break
        
        if auto_regression_column is None:
            print(f"ERROR: No auto-regression data found for {stock}")
            continue

        actual_column = None
        for actual_col in actual_direction_df.columns:
            if stock in actual_col:
                actual_column = actual_col
                break

        if actual_column is None:
            print(f"ERROR: No actual direction data found for {stock}")
            continue

        correct_stock_predictions = 0
        total_stock_predictions = 0

        data_intersection_dates = auto_regression_direction_df.index.intersection(sentiment_df.index).intersection(actual_direction_df.index) # gets the intersection from the actual, auto regression, and sentiment data .csv files

        for date in data_intersection_dates:
            if pd.isna(auto_regression_direction_df.loc[date, auto_regression_column]) or pd.isna(sentiment_df.loc[date, sentiment_column]) or pd.isna(actual_direction_df.loc[date, actual_column]): # skip any dates for which there is a missing value
                continue

            auto_regression_direction = int(auto_regression_direction_df.loc[date, auto_regression_column]) # Get values for this stock and day
            average_sentiment = float(sentiment_df.loc[date, sentiment_column])
            actual_direction = int(actual_direction_df.loc[date, actual_column])

            combined_model_direction = prediction_for_stock_and_day(auto_regression_direction, average_sentiment, args.base_confidence, args.confidence_threshold)

            if combined_model_direction == actual_direction:
                correct_stock_predictions += 1
                correct_predictions += 1
            
            total_stock_predictions += 1
            total_predictions += 1

        if total_stock_predictions > 0:
            stock_accuracy = correct_stock_predictions / total_stock_predictions
            stock_accuracies[stock] = stock_accuracy
            print(f"{stock}: Accuracy = {stock_accuracy:.4f}")
        
    auto_regression_direction_df.reset_index(inplace=True)
    sentiment_df.reset_index(inplace=True)
    actual_direction_df.reset_index(inplace=True)

    if total_predictions > 0:
        accurracy = correct_predictions / total_predictions
        print(f"\nOverall Accuracy: {accurracy:.4f}, Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")

        print(f"\nBase Confidence: {args.base_confidence}, Confidence Threshold: {args.confidence_threshold}")
        print(f"Sentiment Model Used: {args.model_name}")

if __name__ == "__main__":
    main()
    