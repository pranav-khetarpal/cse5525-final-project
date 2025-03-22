from datasets import load_dataset
import pandas as pd
import os

def save_split_to_csv(dataset, split_name, save_dir):
    """
    Save a dataset split as a CSV file.
    
    dataset: Hugging Face dataset split
    split_name: Name of the split (train, validation, test)
    save_dir: Directory where CSV files will be stored
    """
    df = pd.DataFrame(dataset)
    
    # Ensure the tweet folder exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the file
    save_path = os.path.join(save_dir, f"{split_name}_zeroshot.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {split_name} data to {save_path}")

# Load the dataset
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

# Directory to store the files
tweet_dir = "tweet"

# Save each split
for split in ds.keys():  # Keys are train, validation, test
    save_split_to_csv(ds[split], split, tweet_dir)
