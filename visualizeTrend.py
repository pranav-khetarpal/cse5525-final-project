import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from combined_model import (
    prediction_for_stock_and_day,
    BERT_TEST_SENTIMENT_PATH,
    FINANCIALBERT_TEST_SENTIMENT_PATH,
    AUTOREGRESSION_TEST_DIRECTION_PATH,
    ACTUAL_TEST_DIRECTION_PATH,
)

def get_args():
    parser = argparse.ArgumentParser(description="Visualize trend predictions")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="BERT",
        choices=["BERT", "FinancialBERT"],
        help="What sentiment model to use",
    )
    
    parser.add_argument(
        "--base_confidence",
        type=float,
        default=0.65,
        help="Base confidence parameter",
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.10,
        help="Confidence threshold parameter",
    )
    
    parser.add_argument(
        "--boost_confidence",
        type=float,
        default=0.10,
        help="Boost confidence parameter",
    )
    
    parser.add_argument(
        "--stocks",
        type=str,
        nargs="+",
        default=["AAPL", "GOOG", "AMZN", "JPM"],
        help="List of stocks to visualize",
    )
    
    return parser.parse_args()

def load_data(model_name, stocks):
    """Load required data for visualization"""
    # Load sentiment data
    if model_name == "BERT":
        sentiment_df = pd.read_csv(BERT_TEST_SENTIMENT_PATH)
    else:
        sentiment_df = pd.read_csv(FINANCIALBERT_TEST_SENTIMENT_PATH)
    
    # Load auto-regression and actual data
    auto_regression_df = pd.read_csv(AUTOREGRESSION_TEST_DIRECTION_PATH)
    actual_df = pd.read_csv(ACTUAL_TEST_DIRECTION_PATH)
    
    # Set date as index
    auto_regression_df['date'] = pd.to_datetime(auto_regression_df['date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    actual_df['date'] = pd.to_datetime(actual_df['date'])
    
    auto_regression_df.set_index('date', inplace=True)
    sentiment_df.set_index('date', inplace=True)
    actual_df.set_index('date', inplace=True)
    
    # Filter data for selected stocks
    stock_data = {}
    
    for stock in stocks:
        stock_info = {'dates': []}
        
        # Find relevant columns
        sentiment_col = None
        for col in sentiment_df.columns:
            if stock in col:
                sentiment_col = col
                break
        
        auto_regression_col = None
        for col in auto_regression_df.columns:
            if stock in col:
                auto_regression_col = col
                break
        
        actual_col = None
        for col in actual_df.columns:
            if stock in col:
                actual_col = col
                break
        
        if not (sentiment_col and auto_regression_col and actual_col):
            print(f"Skipping {stock} - missing data")
            continue
        
        # Get common dates
        common_dates = auto_regression_df.index.intersection(
            sentiment_df.index).intersection(actual_df.index)
        
        stock_info['dates'] = common_dates
        stock_info['sentiment'] = sentiment_df.loc[common_dates, sentiment_col]
        stock_info['regression'] = auto_regression_df.loc[common_dates, auto_regression_col]
        stock_info['actual'] = actual_df.loc[common_dates, actual_col]
        
        stock_data[stock] = stock_info
    
    return stock_data

def generate_predictions(stock_data, base_confidence, confidence_threshold, boost_confidence):
    """Generate combined model predictions for each stock"""
    for stock, data in stock_data.items():
        predictions = []
        correct_predictions = 0
        
        for i, date in enumerate(data['dates']):
            if (pd.isna(data['regression'].iloc[i]) or 
                pd.isna(data['sentiment'].iloc[i]) or 
                pd.isna(data['actual'].iloc[i])):
                predictions.append(np.nan)
                continue
            
            auto_regression_direction = int(data['regression'].iloc[i])
            average_sentiment = float(data['sentiment'].iloc[i])
            actual_direction = int(data['actual'].iloc[i])
            
            prediction = prediction_for_stock_and_day(
                auto_regression_direction,
                average_sentiment,
                base_confidence,
                confidence_threshold,
                boost_confidence
            )
            
            predictions.append(prediction)
            
            if prediction == actual_direction:
                correct_predictions += 1
        
        total_valid = sum(~pd.isna(predictions))
        accuracy = correct_predictions / total_valid if total_valid > 0 else 0
        
        # Store predictions and metrics
        data['predictions'] = predictions
        data['accuracy'] = accuracy
        data['correct_count'] = correct_predictions
        data['total_valid'] = total_valid
    
    return stock_data

def visualize_trends(stock_data, model_name):
    """Visualize actual vs predicted trends for each stock"""
    stocks = list(stock_data.keys())
    n_stocks = len(stocks)
    
    # Create a figure with a subplot for each stock
    fig, axes = plt.subplots(n_stocks, 1, figsize=(14, 5*n_stocks))
    if n_stocks == 1:
        axes = [axes]
    
    # Set overall title
    fig.suptitle(f'Stock Trend Direction Prediction ({model_name} Model)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for i, stock in enumerate(stocks):
        data = stock_data[stock]
        dates = data['dates']
        actual = data['actual']
        predictions = data['predictions']
        
        ax = axes[i]
        
        # Plot actual direction as a line
        ax.plot(dates, actual, 'b-', label='Actual Direction', linewidth=2)
        
        # Plot predictions as scatter points
        correct_dates = []
        incorrect_dates = []
        correct_preds = []
        incorrect_preds = []
        
        for j, date in enumerate(dates):
            if pd.isna(predictions[j]) or pd.isna(actual.iloc[j]):
                continue
                
            if predictions[j] == actual.iloc[j]:
                correct_dates.append(date)
                correct_preds.append(predictions[j])
            else:
                incorrect_dates.append(date)
                incorrect_preds.append(predictions[j])
        
        # Plot correct and incorrect predictions
        ax.scatter(correct_dates, correct_preds, color='green', s=80, 
                   label='Correct Prediction', zorder=3, alpha=0.7)
        ax.scatter(incorrect_dates, incorrect_preds, color='red', s=80, 
                   label='Incorrect Prediction', zorder=3, alpha=0.7)
        
        # Add a background color to better distinguish up (1) from down (0)
        ax.axhspan(0.5, 1.1, facecolor='#e6f7e6', alpha=0.3)  # Light green for up
        ax.axhspan(-0.1, 0.5, facecolor='#f7e6e6', alpha=0.3)  # Light red for down
        
        # Set y-axis limits and labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Down', 'Up'])
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add title with accuracy information
        ax.set_title(f"{stock}: Prediction Accuracy = {data['accuracy']:.2f} ({data['correct_count']}/{data['total_valid']})",
                     fontsize=14, fontweight='bold')
        
        # Add labels
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Trend Direction', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(f'trend_prediction_visualization_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'trend_prediction_visualization_{model_name}.pdf', format='pdf', bbox_inches='tight')
    
    print(f"Visualization saved as trend_prediction_visualization_{model_name}.png and .pdf")
    
    return fig

def main():
    args = get_args()
    
    print(f"Visualizing trends for: {', '.join(args.stocks)}")
    print(f"Using {args.model_name} model with parameters:")
    print(f"  Base Confidence: {args.base_confidence}")
    print(f"  Confidence Threshold: {args.confidence_threshold}")
    print(f"  Boost Confidence: {args.boost_confidence}")
    
    # Load data
    stock_data = load_data(args.model_name, args.stocks)
    
    if not stock_data:
        print("No valid stock data found. Check stock symbols and data files.")
        return
    
    # Generate predictions
    stock_data = generate_predictions(
        stock_data, 
        args.base_confidence, 
        args.confidence_threshold, 
        args.boost_confidence
    )
    
    # Create visualization
    visualize_trends(stock_data, args.model_name)
    
    # Print summary
    print("\nPrediction Summary:")
    for stock, data in stock_data.items():
        print(f"{stock}: Accuracy = {data['accuracy']:.4f} ({data['correct_count']}/{data['total_valid']})")

if __name__ == "__main__":
    args = get_args()
    main()