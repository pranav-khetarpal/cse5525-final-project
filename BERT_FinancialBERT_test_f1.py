import os
import argparse
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BERT_PATH = "./BERT/bert_model"
FINANCIAL_BERT_PATH = "./FinancialBert/financial_bert_model"
TEST_DATA = "./combined_model_data/processed_test_stockemo_with_dates.csv"

BEST_TRAINED_WEIGHTS_PATH_BERT = "BERT_experiment_39.pt"
BEST_TRAINED_WEIGHTS_PATH_FINANCIALBERT = "FinancialBERT_experiment_13.pt"


def get_args():
    """
    Arguments.
    """
    parser = argparse.ArgumentParser(description="test evalution")

    parser.add_argument(
        "--model_name",
        type=str,
        default="BERT",
        choices=["BERT", "FinancialBERT"],
        help="What model to run results on",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Where to save the prediction results",
    )

    args = parser.parse_args()
    return args


def load_test_data(args) -> DataLoader:
    """
    Load the data from the test CSV file.
    """

    print("inside of load_data")

    def process_file(tokenizer, file_path):
        """
        Process a single file and return a TensorDataset.
        """
        print(f"Processing file: {file_path}")
        data_frame = pd.read_csv(file_path)

        # Verify required columns exist
        required_columns = ["cleaned_tweet", "senti_label"]
        missing_columns = [
            col for col in required_columns if col not in data_frame.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Map text labels to numeric values
        label_map = {"bearish": 0, "bullish": 1}

        # Validate sentiment labels
        invalid_labels = data_frame[~data_frame["senti_label"].isin(label_map.keys())]
        if not invalid_labels.empty:
            print(
                f"\nWarning: Found {len(invalid_labels)} rows with invalid sentiment labels:"
            )
            print(invalid_labels[["cleaned_tweet", "senti_label"]].head())
            print("\nRemoving invalid rows...")
            data_frame = data_frame[data_frame["senti_label"].isin(label_map.keys())]

        labels = [label_map[label] for label in data_frame["senti_label"]]

        dictionary = tokenizer(
            data_frame["cleaned_tweet"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        dataset = TensorDataset(
            dictionary["input_ids"],
            dictionary["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )

        print(f"Successfully loaded {len(dataset)} examples from {file_path}")
        return dataset, data_frame

    tokenizer = BertTokenizer.from_pretrained(
        BERT_PATH
    )  # tokenizer should not matter whether we are using BERT or FinancialBERT

    test_dataset, test_data_frame = process_file(
        tokenizer=tokenizer, file_path=TEST_DATA
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    return test_loader, test_data_frame


def initialize_model(args) -> BertForSequenceClassification:
    """
    Initialize the proper model, either BERT or FinancialBERT.
    """
    if args.model_name == "BERT":
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=BERT_PATH, num_labels=2
        )  # 2 labels for binary classification

        model.to(DEVICE)
        return model
    elif args.model_name == "FinancialBERT":
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=FINANCIAL_BERT_PATH, num_labels=2
        )  # 2 labels for binary classification

        model.to(DEVICE)
        return model
    else:
        print("ERROR: Model name chosen is not one of BERT or FinancialBERT.")


def load_best_model_from_checkpoint(args):
    """
    Load the best BERT or FinancialBERT model from the checkpoint folder
    """
    model_name = args.model_name

    # Set checkpoint directory and filename based on model type
    if model_name == "BERT":
        checkpoint_dir = os.path.join("checkpoint", "BERT_experiments")
        best_file_name = BEST_TRAINED_WEIGHTS_PATH_BERT
    else:  # FinancialBERT
        checkpoint_dir = os.path.join("checkpoint", "FinancialBERT_experiments")
        best_file_name = BEST_TRAINED_WEIGHTS_PATH_FINANCIALBERT

    path = os.path.join(checkpoint_dir, best_file_name)

    # Initialize the base model
    model = initialize_model(args)

    # Check if checkpoint exists and load it
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=DEVICE)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # If checkpoint is a dictionary with model_state_dict
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # If checkpoint is just the state dict
                model.load_state_dict(checkpoint)

            print(f"Successfully loaded model from {path}")
        except Exception as e:
            print(f"ERROR: Could not load model from {path}")
            print(f"Exception: {e}")
    else:
        print(f"ERROR: The checkpoint file does not exist: {path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available files in checkpoint dir:")
        if os.path.exists(os.path.dirname(path)):
            print(os.listdir(os.path.dirname(path)))

    # Move model to the appropriate device
    model.to(DEVICE)

    return model


def predict_tweet_tweet(model, test_loader, test_data_frame, args):
    """
    Generate predictions for test tweets and save to CSV
    """
    # Set up containers for predictions
    all_predictions = []
    all_labels = []

    # Get ticker and date info from dataframe
    tickers = (
        test_data_frame["ticker"].tolist()
        if "ticker" in test_data_frame.columns
        else ["UNKNOWN"] * len(test_data_frame)
    )
    dates = (
        test_data_frame["date"].tolist()
        if "date" in test_data_frame.columns
        else ["UNKNOWN"] * len(test_data_frame)
    )
    tweets = test_data_frame["cleaned_tweet"].tolist()

    # Run prediction
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predictions (0=bearish, 1=bullish)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # Create prediction dataframe
    results_df = pd.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "tweet": tweets,
            "true_label": test_data_frame["senti_label"].tolist(),
            "prediction": ["bullish" if p == 1 else "bearish" for p in all_predictions],
            "prediction_numeric": all_predictions,
        }
    )

    # Calculate accuracy and F1 score
    accuracy = (
        results_df["prediction_numeric"]
        == [1 if l == "bullish" else 0 for l in results_df["true_label"]]
    ).mean()

    print(f"Test Accuracy: {accuracy:.4f}")

    from sklearn.metrics import f1_score

    actual_numerical_classes = [1 if label == "bullish" else 0 for label in results_df["true_label"]]

    f1 = f1_score(
            actual_numerical_classes,
            results_df["prediction_numeric"],
            average="weighted",
        )
    
    print(f"Test F1 Score: {f1:.4f}")

    # # Save predictions to CSV
    # results_df.to_csv(args.output_csv, index=False)
    # print(f"Saved predictions to {args.output_csv}")

    # # Create the ticker-date map structure
    # sentiment_map = {}

    # for _, row in results_df.iterrows():
    #     ticker = row["ticker"]
    #     date = row["date"]
    #     pred = row["prediction_numeric"]

    #     # Initialize ticker if not exists
    #     if ticker not in sentiment_map:
    #         sentiment_map[ticker] = {}

    #     # Initialize date if not exists
    #     if date not in sentiment_map[ticker]:
    #         sentiment_map[ticker][date] = [0, 0]  # [total_score, count]

    #     # Add prediction to total and increment count
    #     sentiment_map[ticker][date][0] += pred
    #     sentiment_map[ticker][date][1] += 1

    # return sentiment_map


def save_average_sentiment_to_csv(args, sentiment_map):
    """
    """

    from collections import defaultdict

    missing_key_dictionary = defaultdict(dict)

    average_sentiment_csv = f"combined_model_data/{args.model_name}_average_sentiment_per_stock.csv"

    for (ticker, date_dictionary) in sentiment_map.items(): # each ticker and corresponding date dictionary
        for date, [total_score, count] in date_dictionary.items(): # each date and corresponding total tweets and scores
            average_score = 0 # compute average score
            if count > 0:
                average_score = total_score / count
            missing_key_dictionary[date][f"average_sentiment_score_{ticker}"] = average_score # add to dictionary for date, the map for the ticker and average scores, with some tickers missing

    dataframe = pd.DataFrame.from_dict(missing_key_dictionary, orient="index") # keys of missing_key_dictionary should be rows
    dataframe.index.name = "date"
    dataframe = dataframe.sort_index()
    dataframe = dataframe.reset_index()  # saving 'date' as a column as well
    dataframe.to_csv(average_sentiment_csv, index=False)
    print(f"Saved average sentiment scores to {average_sentiment_csv}")


def main():
    # Get key arguments
    args = get_args()

    # Add batch_size for test_loader
    args.batch_size = 16  # Add default batch size

    # Get the Dataloader for the test set
    test_loader, test_data_frame = load_test_data(args=args)

    # Initialize and load the model
    model = initialize_model(args)
    model = load_best_model_from_checkpoint(args)
    model.eval()

    # Generate predictions and create sentiment map
    predict_tweet_tweet(model, test_loader, test_data_frame, args)

    # Save sentiment map to a file
    # map_file = f"{args.model_name}_sentiment_map.json"

    # # Convert to serializable format
    # import json

    # with open(map_file, "w") as f:
    #     json.dump(sentiment_map, f, indent=2)

    # print(f"Saved sentiment map to {map_file}")

    # save_average_sentiment_to_csv(args, sentiment_map)
    # print(f"Saved average sentiment scores")


if __name__ == "__main__":
    main()

