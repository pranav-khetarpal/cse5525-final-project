import re
import pandas as pd
import os

def clean_tweet(tweet):
    # Remove stock tickers (e.g., $TSLA)
    tweet = re.sub(r'\$\w+', '', tweet)
    
    # Remove emojis and special characters
    tweet = re.sub(r'[^\w\s.,!?]', '', tweet)
    
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

def preprocess_tweet_data_from_csv(csv_file, save_path):
    """
    csv_file: the path to the .csv file with the tweet data
    save_path: the path where the cleaned .csv file will be saved
    """
    # Load data
    df = pd.read_csv(csv_file, header=None, names=['id', 'date', 'ticker', 'emo_label', 'senti_label', 'original', 'processed'])

    # Apply cleaning function
    df['cleaned_tweet'] = df['original'].apply(clean_tweet)  # Fixed column reference

    # Select relevant columns
    df = df[['cleaned_tweet', 'senti_label']]  # Ensure correct column selection

    # Save processed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    df.to_csv(save_path, index=False)

    return df

# Example usage
csv_file = "tweet/val_stockemo.csv"  # Input CSV file path
save_path = "tweet/processed_val_stockemo.csv"  # Save cleaned tweets in the tweet folder

processed_data = preprocess_tweet_data_from_csv(csv_file, save_path)
print(processed_data.head())
