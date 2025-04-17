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
    
    
    
def main():
    """
    Combine auto-regression predictions with sentiment analysis and evaluate against actual data
    """
    args = get_args()
    
    # Load sentiment data
    if args.model_name == "BERT":
        sentiment_df = pd.read_csv(BERT_TEST_SENTIMENT_PATH)
    else:
        sentiment_df = pd.read_csv(FINANCIALBERT_TEST_SENTIMENT_PATH)
    
    # Load auto-regression predictions and actual directions
    auto_regression_df = pd.read_csv(AUTOREGRESSION_TEST_DIRECTION_PATH)
    actual_df = pd.read_csv(ACTUAL_TEST_DIRECTION_PATH)
    
    # Prepare results dataframe
    results_df = pd.DataFrame(index=auto_regression_df['date'])
    
    # Track metrics
    correct_predictions = 0
    total_predictions = 0
    stock_accuracies = {}
    
    # Set date as index for easier alignment
    auto_regression_df.set_index('date', inplace=True)
    sentiment_df.set_index('date', inplace=True)
    actual_df.set_index('date', inplace=True)
    
    # Process each stock from sentiment data
    for sentiment_col in sentiment_df.columns:
        if sentiment_col == 'date':
            continue
            
        # Extract stock symbol from column name
        stock = sentiment_col.split('_')[0]  # Assuming format is like "AAPL_sentiment"
        
        # Find corresponding auto-regression column
        regression_col = None
        for reg_col in auto_regression_df.columns:
            if stock in reg_col:
                regression_col = reg_col
                break
                
        if not regression_col:
            print(f"No auto-regression data found for {stock}, skipping...")
            continue
            
        # Find actual direction column
        actual_col = None
        for a_col in actual_df.columns:
            if stock in a_col:
                actual_col = a_col
                break
                
        if not actual_col:
            print(f"No actual direction data found for {stock}, skipping...")
            continue
            
        # Process each day for this stock
        stock_correct = 0
        stock_total = 0
        
        combined_predictions = []
        
        # Get common dates
        common_dates = auto_regression_df.index.intersection(sentiment_df.index).intersection(actual_df.index)
        
        for date in common_dates:
            # Skip if any data is missing
            if pd.isna(auto_regression_df.loc[date, regression_col]) or pd.isna(sentiment_df.loc[date, sentiment_col]) or pd.isna(actual_df.loc[date, actual_col]):
                continue
                
            # Get values for this stock and day
            regression_direction = int(auto_regression_df.loc[date, regression_col])
            average_sentiment = float(sentiment_df.loc[date, sentiment_col])
            actual_direction = int(actual_df.loc[date, actual_col])
            
            # Combine predictions
            final_prediction = confidence_for_stock_and_day(
                regression_direction, 
                average_sentiment, 
                args.base_confidence, 
                args.confidence_threshold
            )
            
            # Store the combined prediction
            combined_predictions.append(final_prediction)
            
            # Check if prediction is correct
            is_correct = (final_prediction == actual_direction)
            if is_correct:
                correct_predictions += 1
                stock_correct += 1
            
            total_predictions += 1
            stock_total += 1
                
        # Calculate accuracy for this stock
        if stock_total > 0:
            stock_accuracy = stock_correct / stock_total
            stock_accuracies[stock] = stock_accuracy
            print(f"{stock}: Accuracy = {stock_accuracy:.4f} ({stock_correct}/{stock_total})")
        
        # Add combined predictions to results dataframe
        results_col_name = f"{stock}_combined_prediction"
        results_df[results_col_name] = pd.Series(combined_predictions, index=common_dates)
    
    # Reset indices for continued processing
    auto_regression_df.reset_index(inplace=True)
    sentiment_df.reset_index(inplace=True)
    actual_df.reset_index(inplace=True)
    
    # Calculate and print overall accuracy
    if total_predictions > 0:
        overall_accuracy = correct_predictions / total_predictions
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # Calculate average stock accuracy
        avg_stock_accuracy = sum(stock_accuracies.values()) / len(stock_accuracies)
        print(f"Average Stock Accuracy: {avg_stock_accuracy:.4f}")
        
        # Print regression vs combined model comparison
        print(f"\nBase Confidence: {args.base_confidence}, Confidence Threshold: {args.confidence_threshold}")
        print(f"Sentiment Model Used: {args.model_name}")
    
    # Save combined predictions to CSV
    output_path = f"combined_model_data/combined_predictions_{args.model_name}_{args.base_confidence}_{args.confidence_threshold}.csv"
    results_df.to_csv(output_path)
    print(f"\nCombined predictions saved to: {output_path}")