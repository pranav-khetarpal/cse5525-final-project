import pandas as pd

# Input and output file paths
input_file = "tweet/VAR_combined_tweet_data.csv"
output_file = "tweet/VAR_chronological_combined_tweet_data.csv"

def sort_csv_by_date(input_path, output_path):
    # Load the CSV file
    df = pd.read_csv(input_path)

    # Convert 'date' column to datetime format (for proper sorting)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows where date couldn't be parsed
    df = df.dropna(subset=['date'])

    # Sort by date in ascending order
    df = df.sort_values(by='date', ascending=True)

    # Write to output CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote sorted data to {output_path}")

# Run the script
if __name__ == "__main__":
    sort_csv_by_date(input_file, output_file)
