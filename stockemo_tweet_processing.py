import re
import pandas as pd
import os

def clean_tweet(tweet):
    # Remove only the first stock ticker (e.g., $TSLA at the beginning)
    tweet = re.sub(r'^\$\w+\s*', '', tweet, count=1)

    # Remove emojis and special characters
    tweet = re.sub(r"[^\w\s.,!?\'’$-]", '', tweet)

    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

def preprocess_tweet_data_from_csv(csv_file, save_path):
    """
    csv_file: the path to the .csv file with the tweet data
    save_path: the path where the cleaned .csv file will be saved
    """
    # Load data
    df = pd.read_csv(csv_file, skiprows=1, header=None, names=['id', 'date', 'ticker', 'emo_label', 'senti_label', 'original', 'processed'])

    # Apply cleaning function
    df['cleaned_tweet'] = df['original'].apply(clean_tweet)  # Fixed column reference

    # Select relevant columns with ticker as the first column
    df = df[['ticker', 'cleaned_tweet', 'senti_label']]  # Ensure correct column selection

    # Save processed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    df.to_csv(save_path, index=False)

    return df

# Example usage
# csv_file = "tweet/train_stockemo.csv"  # Input CSV file path
# save_path = "tweet/processed_train_stockemo.csv"  # Save cleaned tweets in the tweet folder

# processed_data = preprocess_tweet_data_from_csv(csv_file, save_path)
# print(processed_data.head())

# csv_file = "tweet/val_stockemo.csv"  # Input CSV file path
# save_path = "tweet/processed_val_stockemo.csv"  # Save cleaned tweets in the tweet folder

# processed_data = preprocess_tweet_data_from_csv(csv_file, save_path)
# print(processed_data.head())

csv_file = "tweet/test_stockemo.csv"  # Input CSV file path
save_path = "tweet/processed_test_stockemo.csv"  # Save cleaned tweets in the tweet folder

processed_data = preprocess_tweet_data_from_csv(csv_file, save_path)
print(processed_data.head())
