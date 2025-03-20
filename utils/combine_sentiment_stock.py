import pandas as pd
import os
from datetime import datetime

# Parameters
TICKER = "AAPL"
sentiment_file = "tweet/processed_stockemo.csv"
stock_file = f"data/{TICKER}.csv"
output_file = f"data/{TICKER}_price_sentiment.csv"

print(f"Combining tweet sentiment data with stock price data...")

# Load processed tweet data
tweets_df = pd.read_csv(sentiment_file)
print(f"Loaded {len(tweets_df)} tweets with sentiment from {sentiment_file}")

# Check if stock price data exists
if not os.path.exists(stock_file):
    print(f"Error: Stock price data {stock_file} not found!")
    exit(1)

# Load stock price data
stock_df = pd.read_csv(stock_file)
print(f"Loaded {len(stock_df)} days of stock price data from {stock_file}")

# Convert dates to datetime for merging
tweets_df["Date"] = pd.to_datetime(tweets_df["Date"])
stock_df["date"] = pd.to_datetime(stock_df["date"])

# Convert sentiments to numeric values
sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
tweets_df["sentiment_value"] = tweets_df["sentiment"].map(sentiment_map)

# Calculate daily sentiment metrics
print("Calculating daily sentiment metrics...")
daily_sentiment = (
    tweets_df.groupby("Date")
    .agg(
        avg_sentiment=("sentiment_value", "mean"),
        pos_tweets=("sentiment_value", lambda x: sum(x > 0)),
        neg_tweets=("sentiment_value", lambda x: sum(x < 0)),
        neutral_tweets=("sentiment_value", lambda x: sum(x == 0)),
        total_tweets=("sentiment_value", "count"),
    )
    .reset_index()
)

# Merge stock data with sentiment data
print("Merging stock and sentiment data...")
combined_df = pd.merge(
    stock_df, daily_sentiment, left_on="date", right_on="Date", how="left"
)

# Clean up and format the combined data
combined_df.drop("Date", axis=1, inplace=True)  # Remove duplicate date column
combined_df.rename(columns={"date": "Date"}, inplace=True)  # Standardize column name

# Fill NaN values for days without tweets
combined_df["total_tweets"] = combined_df["total_tweets"].fillna(0)
combined_df["avg_sentiment"] = combined_df["avg_sentiment"].fillna(0)
combined_df["pos_tweets"] = combined_df["pos_tweets"].fillna(0)
combined_df["neg_tweets"] = combined_df["neg_tweets"].fillna(0)
combined_df["neutral_tweets"] = combined_df["neutral_tweets"].fillna(0)

# Save the combined data
combined_df.to_csv(output_file, index=False)
print(f"Successfully combined stock price and sentiment data! Saved to {output_file}")

# Basic statistics
print("\nBasic sentiment statistics:")
print(f"Average sentiment: {combined_df['avg_sentiment'].mean():.4f}")
print(f"Days with positive sentiment: {sum(combined_df['avg_sentiment'] > 0)}")
print(f"Days with negative sentiment: {sum(combined_df['avg_sentiment'] < 0)}")
print(f"Days with neutral sentiment: {sum(combined_df['avg_sentiment'] == 0)}")

# Show date range of the data
date_range = f"{combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}"
print(f"\nData covers range: {date_range}")
print(f"Total trading days: {len(combined_df)}")
print(f"Days with tweet data: {sum(combined_df['total_tweets'] > 0)}")

print("\nCombination complete!")
