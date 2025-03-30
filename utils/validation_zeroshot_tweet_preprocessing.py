import pandas as pd
import os
import string
import re

def clean_tweet_text(text):
    # Remove links
    text = re.sub(r'https?://\S+', '', text)

    # Remove leading junk characters (dashes, bullets, colons, etc.)
    text = text.lstrip("–—-:•. ").strip()

    # Normalize whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    return text

# def extract_ticker_and_text(row):
#     words = row['text'].split()
#     tickers = []
#     i = 0

#     # Grab tickers at the start of tweet
#     while i < len(words) and words[i].startswith('$'):
#         raw_ticker = words[i][1:]
#         cleaned_ticker = raw_ticker.translate(str.maketrans('', '', string.punctuation))
#         if cleaned_ticker:
#             tickers.append(cleaned_ticker)
#         i += 1

#     tweet_text = ' '.join(words[i:])
#     tweet_text = clean_tweet_text(tweet_text)

#     return [{'ticker': ticker, 'text': tweet_text, 'label': row['label']} for ticker in tickers]

def extract_ticker_and_text(row):
    text = row['text']
    label = row['label']
    
    words = text.split()
    tickers = []
    i = 0

    while i < len(words) and words[i].startswith('$'):
        raw_ticker = words[i][1:]
        cleaned_ticker = raw_ticker.translate(str.maketrans('', '', string.punctuation))
        if cleaned_ticker:
            tickers.append(cleaned_ticker)
        i += 1

    tweet_text = ' '.join(words[i:]) if tickers else text
    tweet_text = clean_tweet_text(tweet_text)

    if tickers:
        return [{'ticker': ticker, 'text': tweet_text, 'label': label} for ticker in tickers]
    else:
        return [{'ticker': 'empty', 'text': tweet_text, 'label': label}]

def process_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    expanded_rows = []

    for _, row in df.iterrows():
        expanded_rows.extend(extract_ticker_and_text(row))

    df_expanded = pd.DataFrame(expanded_rows)
    df_expanded.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")

# File paths
input_files = {
    "train_zeroshot.csv": "processed_train_zeroshot.csv",
    "validation_zeroshot.csv": "processed_validation_zeroshot.csv"
}

csv_dir = "tweet"

# Run processing
for input_file, output_file in input_files.items():
    input_path = os.path.join(csv_dir, input_file)
    output_path = os.path.join(csv_dir, output_file)
    process_csv(input_path, output_path)