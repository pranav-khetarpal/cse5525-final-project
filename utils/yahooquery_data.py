import os
import pandas as pd
import time
from datetime import datetime
from yahooquery import Ticker
import sys

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# List of tickers to download
tickers = [
    "^GSPC",
    "AAPL",
    "ABNB",
    "AMT",
    "AMZN",
    "BA",
    "BABA",
    "BAC",
    "BKNG",
    "BRK-B",
    "CCL",
    "CVX",
    "DIS",
    "META",
    "GOOG",
    "GOOGL",
    "HD",
    "JNJ",
    "JPM",
    "KO",
    "LOW",
    "MA",
    "MCD",
    "MSFT",
    "NFLX",
    "NKE",
    "NVDA",
    "PFE",
    "PG",
    "PYPL",
    "SBUX",
    "TM",
    "TSLA",
    "TSM",
    "UNH",
    "UPS",
    "V",
    "WMT",
    "XOM",
]


start_date = "2010-01-01"
# end_date = datetime.now().strftime("%Y-%m-%d")
end_date = "2020-01-01"

print(f"Downloading data for {len(tickers)} tickers")
print(f"Date range: {start_date} to {end_date}")


def download_ticker(ticker_symbol):
    try:
        print(f"Downloading {ticker_symbol}...")

        # Handle ticker symbol for filename
        filename = ticker_symbol.replace("^", "").replace("-", "_")
        filepath = f"data/{filename}.csv"

        # Skip if file exists
        if os.path.exists(filepath):
            print(f"  File exists: {filepath}")
            return True

        # Create Ticker object
        ticker = Ticker(ticker_symbol)

        # Download historical data
        data = ticker.history(start=start_date, end=end_date)

        # Check if we have data
        if data.empty:
            print(f"  No data available for {ticker_symbol}")
            return False

        # Reset index if it's a multi-index (multiple symbols)
        if isinstance(data.index, pd.MultiIndex):
            # If we have multiple tickers, filter to just this one
            if len(data.index.levels[0]) > 1:
                data = data.loc[ticker_symbol]
            else:
                data = data.reset_index(level=0, drop=True)

        # Save to CSV
        data.to_csv(filepath)
        print(f"  Saved {filepath} with {len(data)} rows")
        return True

    except Exception as e:
        print(f"  Error downloading {ticker_symbol}: {str(e)[:150]}")
        return False


# Track results
successful = []
failed = []

# Download each ticker with a delay between attempts(2 seconds, should be enough, I think, i am not sure, we'll see )
for i, ticker_symbol in enumerate(tickers):
    print(f"\n[{i+1}/{len(tickers)}] Processing {ticker_symbol}")
    if download_ticker(ticker_symbol):
        successful.append(ticker_symbol)
    else:
        failed.append(ticker_symbol)

    # delay to avoid rate limiting
    if i < len(tickers) - 1:
        wait_time = 2
        print(f"  Waiting {wait_time} seconds before the next ticker...")
        time.sleep(wait_time)

# Print summary
print("\n" + "=" * 50)
print("Download Complete")
print(f"Success: {len(successful)}/{len(tickers)}")
print(f"Failed: {len(failed)}/{len(tickers)}")
if failed:
    print("Failed tickers:", ", ".join(failed))
print("=" * 50)
