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
# DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

BERT_PATH = "./BERT/bert_model"
FINANCIAL_BERT_PATH = "./FinancialBert/financial_bert_model"

TRAIN_DATA = "./tweet/final_processed_merged_stockemo_zeroshot_train.csv"
VAL_DATA = "./tweet/final_processed_merged_stockemo_zeroshot_val.csv"
TEST_DATA = "./tweet/processed_test_stockemo.csv"


def get_args():
    """
    Arguments for training.
    """
    parser = argparse.ArgumentParser(description="BERT training loop")

    parser.add_argument(
        "--model_name",
        type=str,
        default="BERT",
        choices=["BERT", "FinancialBERT"],
        help="What model to fine-tune",
    )

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="Whether to use a LR scheduler and what type to use if so",
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=0,
        help="How many epochs to warm up the learning rate for if using a scheduler",
    )
    parser.add_argument(
        "--max_n_epochs",
        type=int,
        default=0,
        help="How many epochs to train the model for",
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=0,
        help="If validation performance stops improving, how many epochs should we wait before stopping?",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="How should we name this experiment?",
    )

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def load_data(args) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the data from the respective CSV file.
    """

    print("inside of load_data")

    def process_file(tokenizer, file_path):
        """
        Process a single file and return a TensorDataset.
        """
        print(f"Processing file: {file_path}")
        data_frame = pd.read_csv(file_path)
        initial_size = len(data_frame)

        # Verify required columns exist
        required_columns = ["cleaned_tweet", "senti_label"]
        missing_columns = [
            col for col in required_columns if col not in data_frame.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Map text labels to numeric values
        label_map = {"bearish": 0, "bullish": 1}

        # Validate sentiment labels and count invalid ones
        invalid_labels = data_frame[~data_frame["senti_label"].isin(label_map.keys())]
        if not invalid_labels.empty:
            print(
                f"\nWarning: Found {len(invalid_labels)} rows with invalid sentiment labels"
            )
            print("\nInvalid label distribution:")
            invalid_label_counts = invalid_labels["senti_label"].value_counts()
            for label, count in invalid_label_counts.items():
                print(f"  {label}: {count} rows")

            print("\nExample rows with invalid labels:")
            print(invalid_labels[["cleaned_tweet", "senti_label"]].head())

            # Remove invalid rows
            data_frame = data_frame[data_frame["senti_label"].isin(label_map.keys())]
            final_size = len(data_frame)
            print(f"\nRemoved {initial_size - final_size} rows with invalid labels")
            print(f"Remaining valid label distribution:")
            print(data_frame["senti_label"].value_counts())

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

        print(f"\nSuccessfully loaded {len(dataset)} examples from {file_path}")
        return dataset

    tokenizer = BertTokenizer.from_pretrained(
        BERT_PATH
    )  # tokenizer should not matter whether we are training BERT or FinancialBERT

    train_dataset = process_file(
        tokenizer=tokenizer, file_path=TRAIN_DATA
    )  # getting TensorDatasets, which is basically like Dataset
    val_dataset = process_file(tokenizer=tokenizer, file_path=VAL_DATA)
    test_dataset = process_file(tokenizer=tokenizer, file_path=TEST_DATA)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )  # getting DataLoaders, only shuffle in training
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


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


def initialize_optimizer_and_scheduler(
    args, model, epoch_length
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """ """

    def get_parameter_names(model, forbidden_layer_types):
        """
        Getting all the names of the model's parameters, with no forbidden layer types.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]

        result += list(model._parameters.keys())
        return result

    # Get parameters that should have weight decay
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name
    ]  # decay parameters do not involve 'bias'

    # Group parameters for optimizer
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if n in decay_parameters
            ],  # list of decay_paramters that we want to do weight_decay on
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],  # list of decay_paramters that we do NOT want to do weight_decay on
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        scheduler = None
    elif args.scheduler_type == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif args.scheduler_type == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def load_model_from_checkpoint(args, best):
    """ """
    model_name = args.model_name
    checkpoint_dir = os.path.join(
        "checkpoint", f"{model_name}_experiments"
    )  # create folder for BERT or FinancialBERT
    best_or_last_file_name = "best_model.pt" if best else "last_model.pt"
    path = os.path.join(
        checkpoint_dir, best_or_last_file_name
    )  # get .pt of the best model or last model

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


def save_model(checkpoint_dir, model, best):
    """
    checkpoint_dir: the folder of where to the save the model
    """

    os.makedirs(
        checkpoint_dir, exist_ok=True
    )  # automatically create the directory if it does not exist, otherwise, treat it like normal

    if best:
        file_name = "best_model.pt"
    else:
        file_name = "last_model.pt"

    path = os.path.join(checkpoint_dir, file_name)
    torch.save(model.state_dict(), path)
    print(f"\nModel Saved at {path}\n\n")


def train(args, model, train_loader, val_loader, optimizer, scheduler):
    """ """
    print(f"training has started.")

    best_f1 = -1
    epochs_since_improvement = 0

    model_name = args.model_name
    checkpoint_dir = os.path.join(
        "checkpoint", f"{model_name}_experiments"
    )  # create folder for BERT or FinancialBERT

    for epoch in range(args.max_n_epochs):
        average_train_loss, average_train_accurracy = train_epoch(
            args, model, train_loader, optimizer, scheduler
        )
        print(
            f"\nEpoch {epoch}: Average train loss was {average_train_loss:.4f}, Average accurracy was {average_train_accurracy:.4f}\n"
        )

        average_eval_loss, labels_f1 = eval_epoch(args, model, val_loader)
        print(
            f"Average evaluation loss was {average_eval_loss:.4f}, F1 score was {labels_f1:.4f}"
        )

        if labels_f1 > best_f1:  # found a better F1 score than previously found
            best_f1 = labels_f1
            epochs_since_improvement = 0
        else:  # f1 score is worse than best found so far
            epochs_since_improvement += 1

        save_model(
            checkpoint_dir, model, best=False
        )  # save the last checkpoint for the model
        if epochs_since_improvement == 0:  # save the best model checkpoint
            save_model(checkpoint_dir, model, best=True)

        if (
            epochs_since_improvement >= args.patience_epochs
        ):  # patience epochs reached, so stop
            print(f"Early stopping triggered at epoch: {epoch+1}")
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    """ """
    print(f"inside of train_epoch.")

    model.train()
    total_loss = 0
    loss_function = nn.CrossEntropyLoss()

    total_num_labels = 0
    correctly_predicted_labels = 0

    for input_ids, attention_mask, senti_label in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        senti_label = senti_label.to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # using CrossEntropyLoss
        loss = loss_function(logits, senti_label)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += (
            loss.item()
        )  # since we are doing sentiment classification, loss is just the loss on the predicted label (of the batch) ???

        _, predicted = torch.max(logits, 1)
        total_num_labels += senti_label.size(0)
        correctly_predicted_labels += (predicted == senti_label).sum().item()

    average_loss = total_loss / len(train_loader)
    average_accurracy = correctly_predicted_labels / total_num_labels

    return average_loss, average_accurracy


def eval_epoch(args, model, val_loader):
    """ """
    print(f"inside of eval_epoch")

    model.eval()
    total_loss = 0
    loss_function = nn.CrossEntropyLoss()

    actual_numerical_classes = []  # the true, actual classes
    all_predicted_numerical_classes = []  # numerical scores for the predicted classes

    with torch.no_grad():
        for input_ids, attention_mask, senti_label in tqdm(val_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            senti_label = senti_label.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # using CrossEntropyLoss
            loss = loss_function(logits, senti_label)
            total_loss += loss.item()

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


def test(args, model, test_loader):
    """
    Evaluate the model on the test set and print detailed metrics.
    """
    print("\nStarting test evaluation...")

    model.eval()
    total_loss = 0
    loss_function = nn.CrossEntropyLoss()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, senti_label in tqdm(test_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            senti_label = senti_label.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_function(logits, senti_label)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(senti_label.cpu().numpy())

    # Calculate metrics
    average_loss = total_loss / len(test_loader)
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    # Convert numeric predictions back to labels for better interpretability
    label_map = {0: "bearish", 1: "bullish"}
    text_predictions = [label_map[pred] for pred in all_predictions]
    text_labels = [label_map[label] for label in all_labels]

    # Print detailed results
    print("\nTest Results:")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print example predictions
    print("\nSample Predictions (first 10):")
    print("Expected\tPredicted")
    print("-" * 30)
    for true, pred in zip(text_labels[:10], text_predictions[:10]):
        print(f"{true}\t\t{pred}")

    return average_loss, f1


def main():
    # Get key arguments
    args = get_args()

    # get the Dataloader for train, validation, and test set
    train_loader, val_loader, test_loader = load_data(args=args)

    if args.max_n_epochs > 0:  # Only train if epochs > 0
        print("\n=== Starting Training ===")
        model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(
            args, model, len(train_loader)
        )
        train(args, model, train_loader, val_loader, optimizer, scheduler)

    # Load the best model for evaluation
    print("\n=== Loading Best Model for Evaluation ===")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Evaluate on validation set
    print("\n=== Validation Set Evaluation ===")
    average_val_loss, val_f1_score = eval_epoch(args, model, val_loader)
    print(
        f"Validation Results: Average loss was {average_val_loss:.4f}, F1 score was {val_f1_score:.4f}"
    )

    # Test on test set
    print("\n=== Test Set Evaluation ===")
    average_test_loss, test_f1_score = test(args, model, test_loader)
    print(
        f"Test Results: Average loss was {average_test_loss:.4f}, F1 score was {test_f1_score:.4f}"
    )

if __name__ == "__main__":
    main()
