import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Parameters
TICKER = "AAPL"
NUM_TWEETS_PER_DAY = 20
START_DATE = datetime(2021, 1, 4)
END_DATE = datetime(2025, 1, 4)

# Create a directory for data if it doesn't exist
os.makedirs("data", exist_ok=True)

# Sample tweets templates about Apple stock
positive_tweets = [
    "Just bought more $AAPL today. Love this company's long-term prospects!",
    "$AAPL hitting new highs! Best tech stock out there.",
    "Apple's earnings were amazing. $AAPL to the moon!",
    "The new iPhone looks incredible. $AAPL will crush this quarter.",
    "Analysts are underestimating $AAPL growth potential. Strong buy!",
    "Apple's services division is growing faster than expected. Bullish on $AAPL.",
    "$AAPL dividend increase looks promising. Great long-term hold.",
    "Tim Cook's leadership at Apple is outstanding. $AAPL is a winner.",
    "Apple AR headset will be a game changer. $AAPL breaking $200 soon.",
    "Apple car rumors making me even more bullish on $AAPL.",
]

negative_tweets = [
    "$AAPL overvalued at these levels. Taking profits today.",
    "Apple's innovation has stalled under Tim Cook. $AAPL is a sell.",
    "iPhone sales declining in China. Not good for $AAPL.",
    "Competition heating up in wearables. $AAPL losing edge.",
    "$AAPL facing regulatory headwinds. Bearish outlook.",
    "The new MacBooks have serious issues. $AAPL quality declining.",
    "Selling my $AAPL position today. Tech sell-off coming.",
    "Apple's App Store fees under scrutiny. $AAPL faces regulatory risk.",
    "iPhone market share dropping globally. $AAPL growth story ending.",
    "Shorting $AAPL ahead of earnings. Expectations too high.",
]

neutral_tweets = [
    "Holding my $AAPL position for now. Waiting for next earnings.",
    "$AAPL trading sideways today. Market uncertainty.",
    "Apple event announced for next month. $AAPL impact unclear.",
    "Rebalanced my portfolio, keeping same $AAPL allocation.",
    "Interesting patent from Apple today. $AAPL worth watching.",
    "$AAPL volume lower than usual. Market seems indecisive.",
    "Apple suppliers reporting mixed results. Neutral on $AAPL.",
    "Tim Cook sold some shares. Standard procedure for $AAPL execs.",
    "$AAPL trading at average multiple. Fair value currently.",
    "New competitors in Apple's space. $AAPL holding steady though.",
]


# Function to generate random tweets for a date
def generate_tweets(date, num_tweets=NUM_TWEETS_PER_DAY):
    date_str = date.strftime("%Y-%m-%d")
    tweets = []

    # Sentiment distribution: 40% positive, 30% negative, 30% neutral
    sentiment_distribution = ["positive"] * 40 + ["negative"] * 30 + ["neutral"] * 30

    for i in range(num_tweets):
        sentiment = random.choice(sentiment_distribution)

        if sentiment == "positive":
            text = random.choice(positive_tweets)
        elif sentiment == "negative":
            text = random.choice(negative_tweets)
        else:
            text = random.choice(neutral_tweets)

        # Add some randomization to the tweets
        if random.random() > 0.7:
            text += f" #{random.choice(['invest', 'stocks', 'technology', 'trading', 'finance'])}"

        tweet_id = int(date.timestamp() * 1000) + i
        username = f"investor_{random.randint(100, 999)}"
        created_at = (
            date.strftime("%Y-%m-%d")
            + f" {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        )
        retweets = random.randint(0, 100)
        likes = random.randint(5, 500)

        tweets.append(
            {
                "Tweet_ID": tweet_id,
                "Date": date_str,
                "Ticker": TICKER,
                "Username": username,
                "Text": text,
                "Created_At": created_at,
                "Retweets": retweets,
                "Likes": likes,
                "sentiment": sentiment,
            }
        )

    return tweets


# Generate tweets for each day in the date range
all_tweets = []
current_date = START_DATE

print(
    f"Generating synthetic tweet data for {TICKER} from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}"
)

while current_date <= END_DATE:
    # Skip weekends (optional)
    if current_date.weekday() < 5:  # 0-4 are Monday to Friday
        day_tweets = generate_tweets(current_date)
        all_tweets.extend(day_tweets)
        print(
            f"Generated {len(day_tweets)} tweets for {current_date.strftime('%Y-%m-%d')}"
        )

    current_date += timedelta(days=1)

# Create DataFrame
df = pd.DataFrame(all_tweets)
print(f"Total tweets generated: {len(df)}")

# Save full dataset
df.to_csv("processed_stockemo.csv", index=False)
print("Saved processed_stockemo.csv")

# Create train/val/test split (70/15/15)
unique_dates = df["Date"].unique()
num_dates = len(unique_dates)

# Random shuffle dates
np.random.seed(42)
np.random.shuffle(unique_dates)

# Split dates
train_dates = unique_dates[: int(0.7 * num_dates)]
val_dates = unique_dates[int(0.7 * num_dates) : int(0.85 * num_dates)]
test_dates = unique_dates[int(0.85 * num_dates) :]

# Create datasets
train_df = df[df["Date"].isin(train_dates)]
val_df = df[df["Date"].isin(val_dates)]
test_df = df[df["Date"].isin(test_dates)]

# Save splits
train_df.to_csv("train_stockemo.csv", index=False)
val_df.to_csv("val_stockemo.csv", index=False)
test_df.to_csv("test_stockemo.csv", index=False)

print(
    f"Created splits: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples"
)

# Create a sample stock price dataset if it doesn't exist
if not os.path.exists(f"data/{TICKER}.csv"):
    print(f"Creating sample stock price data for {TICKER}")

    # Generate date range
    dates = []
    current_date = START_DATE

    while current_date <= END_DATE:
        if current_date.weekday() < 5:  # Skip weekends
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    # Start with a price and generate random walk
    start_price = 130.0  # Apple's approximate price in early 2021
    price = start_price
    prices = []

    for _ in dates:
        # Random daily change (-2% to +2%)
        daily_change = np.random.normal(0.0005, 0.015)  
        price *= 1 + daily_change
        prices.append(price)

    # Create OHLC data
    stock_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [p * (1 - random.uniform(0, 0.005)) for p in prices],
            "High": [p * (1 + random.uniform(0, 0.01)) for p in prices],
            "Low": [p * (1 - random.uniform(0, 0.01)) for p in prices],
            "Close": prices,
            "Volume": [random.randint(50000000, 200000000) for _ in prices],
        }
    )

    # Save the stock data
    stock_df.to_csv(f"data/{TICKER}.csv", index=False)
    print(f"Saved sample stock price data to data/{TICKER}.csv")

print("All data generation complete!")
