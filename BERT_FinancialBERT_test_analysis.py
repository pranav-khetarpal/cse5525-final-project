import os
import sys
import argparse
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import numpy
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BERT_PATH = "./BERT/bert_model"
FINANCIAL_BERT_PATH = "./FinancialBert/financial_bert_model"

BERT_BEST_MODEL_PATH = "./BERT/bert_model"
FINANCIAL_BERT_BEST_MODEL_PATH = "./FinancialBert/financial_bert_model"

TEST_DATA = "./tweet/processed_test_stockemo.csv"


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

    test_dataset, test_data_frame = process_file(tokenizer=tokenizer, file_path=TEST_DATA)
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
    checkpoint_dir = os.path.join(
        "checkpoint", f"{model_name}_experiments"
    )  # create folder for BERT or FinancialBERT
    best_file_name = "BERT_experiment_39.pt"
    path = os.path.join(
        checkpoint_dir, best_file_name
    )  # get .pt of the best model

    model = initialize_model(args)

    if os.path.exists(path):
        try:
            checkpoint_model_weights = torch.load(path)
            model.load_state_dict(checkpoint_model_weights)
            print(f"Successfully loaded model from {path}")
        except FileExistsError:
            print(f"ERROR: Could not load model from {path}")
    else:
        print(f"ERROR: The path does not exist: {path}")

    model.to(DEVICE)

    return model

def predict_tweet_tweet(model, test_loader, test_data_frame):
    """
    """
    with torch.no_grad():
        for input_ids, attention_mask, senti_label in tqdm(test_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            senti_label = senti_label.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, predicted = torch.max(logits, 1)
            all_predicted_numerical_classes.extend(predicted.cpu().numpy())
            actual_numerical_classes.extend(senti_label.cpu().numpy())

        average_loss = total_loss / len(val_loader)
        f1 = f1_score(
            actual_numerical_classes,
            all_predicted_numerical_classes,
            average="weighted",
        )

        print(f"finished evaluating current epoch")
        return average_loss, f1


def main():
    # Get key arguments
    args = get_args()

    # get the Dataloader for the test set
    test_loader, test_data_frame = load_test_data(args=args)

    model = initialize_model(args)

    # load best model from checkpoint
    model = load_best_model_from_checkpoint(
        args
    )

    model.eval()

    predict_test_tweet(model, test_loader, test_data_frame)


if __name__ == "__main__":
    main()

