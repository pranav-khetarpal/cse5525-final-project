import os
import pandas as pd

#folder containing all {TICKER}.csv files
data_folder = "data/"

# Output file path
output_file = "data/new_merged_closing_prices.csv"

# the column to join on is Date
join_on = "date"

def merge_closing_prices(folder_path, output_path):
    all_closes = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            ticker = os.path.splitext(filename)[0]  # removes '.csv'

            # Skip ABNB
            if ticker.upper() == "ABNB":
                print(f"Skipping {ticker} (excluded ticker)")
                continue

            file_path = os.path.join(folder_path, filename)

            df = pd.read_csv(file_path)

            # Check if 'Close' exists
            if "close" not in df.columns:
                print(f"Skipping {filename}: 'close' column not found.")
                continue

            # If Date exists, use it as index for alignment
            if join_on in df.columns:
                df = df[[join_on, "close"]].copy()
                df.set_index(join_on, inplace=True)
            else:
                df = df[["close"]].copy()

            df.rename(columns={"close": f"{ticker}_close"}, inplace=True)
            all_closes.append(df)

    if not all_closes:
        print("No valid files with 'close' column found.")
        return

    # Merge all DataFrames on index (Date or positional)
    merged_df = pd.concat(all_closes, axis=1)

    merged_df.to_csv(output_path)
    print(f"Successfully wrote merged closing prices to {output_path}")

# Run the script
if __name__ == "__main__":
    merge_closing_prices(data_folder, output_file)
