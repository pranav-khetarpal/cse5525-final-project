import pandas as pd
import os

# Files to clean
file_paths = [
    "tweet/processed_merged_stockemo_zeroshot_train.csv",
    "tweet/processed_merged_stockemo_zeroshot_val.csv"
]

for path in file_paths:
    df = pd.read_csv(path)
    
    # Remove all leading dollar signs from tickers
    df['ticker'] = df['ticker'].apply(lambda x: x.lstrip('$') if isinstance(x, str) else x)

    # Save to new file
    new_path = path.replace("processed_", "final_processed_")
    df.to_csv(new_path, index=False)
    print(f"Cleaned file saved to: {new_path}")