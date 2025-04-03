import os
import sys
import argparse
from typing import Optional
import torch
from torch.utils.data import TensorDataset
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BERT_PATH = './BERT/bert_model'
FINANCIAL_BERT_PATH = './FinancialBert/financial_bert_model'

TRAIN_DATA = './tweet/final_processed_merged_stockemo_zeroshot_train.csv'
VAL_DATA = './tweet/final_processed_merged_stockemo_zeroshot_val.csv'
TEST_DATA = './processed_test_stockemo.csv'

def get_args():
    '''
    Arguments for training.
    '''
    parser = argparse.ArgumentParser(description='BERT training loop')

    parser.add_argument('--model_name', type=str, default="BERT", choices=["BERT, FinancialBERT"], help="What model to fine-tune")

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"], help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0, help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0, help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0, help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--experiment_name', type=str, default='experiment', help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    
    args = parser.parse_args()
    return args


def load_data(args) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the data from the respective CSV file.
    """
    def process_file(tokenizer, file_path):
        """

        """
        data_frame = pd.read_csv(filepath_or_buffer=file_path)

        sentiment_map = {"bearish": 0, "bullish": 1}
        labels = data_frame['senti_label'].map(sentiment_map).values # get the labels for examples

        dictionary = tokenizer(
            data_frame['cleaned_tweet'].tolist(), 
            return_tensors="pt"
        ) # dictionary from which we can extract needed information

        dataset = TensorDataset(
            dictionary['input_ids'],
            dictionary['attention_mask'],
            torch.tensor(labels)
        ) # create TensorDatasets to pass to DataLoaders

        return dataset

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH) # tokenizer should not matter whether we are training BERT or FinancialBERT

    train_dataset = process_file(tokenizer=tokenizer, file_path=TRAIN_DATA) # getting TensorDatasets, which is basically like Dataset
    val_dataset = process_file(tokenizer=tokenizer, file_path=VAL_DATA)
    test_dataset = process_file(tokenizer=tokenizer, file_path=TEST_DATA)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True) # getting DataLoaders, only shuffle in training
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def initialize_model(args) -> BertForSequenceClassification:
    """
    Initialize the proper model, either BERT or FinancialBERT.
    """
    if args.model_name == 'BERT':
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=BERT_PATH, num_labels=2) # 2 labels for binary classification

        model.to(DEVICE)
        return model
    elif args.model_name == 'FinancialBERT':
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=FINANCIAL_BERT_PATH, num_labels=2) # 2 labels for binary classification

        model.to(DEVICE)
        return model
    else:
        print("ERROR: Model name chosen is not one of BERT or FinancialBERT.")


def initialize_optimizer_and_scheduler(args, model, epoch_length) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    """
    
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
    decay_parameters = [name for name in decay_parameters if "bias" not in name] # decay parameters do not involve 'bias'

    # Group parameters for optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters], # list of decay_paramters that we want to do weight_decay on
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters], # list of decay_paramters that we do NOT want to do weight_decay on
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        scheduler = None
    elif args.scheduler_type == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif args.scheduler_type == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        raise NotImplementedError
    
    return optimizer, scheduler


def load_model_from_checkpoint(args, best):
    """
    """
    model_name = args.model_name
    checkpoint_dir = os.path.join("checkpoint", f"{model_name}_experiments") # create folder for BERT or FinancialBERT
    best_or_last_file_name = "best_model.pt" if best else "last_model.pt"
    path = os.path.join(checkpoint_dir, best_or_last_file_name) # get .pt of the best model or last model

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

    if best:
        file_name = "best_model.pt"
    else:
        file_name = "last_model.pt"

    path = os.path.join(checkpoint_dir, file_name)
    torch.save(model.state_dict(), path)
    print(f"\nModel Saved at {path}\n\n")


def train(args, model, train_loader, val_loader, optimizer, scheduler):
    """
    """

def train_epoch(args, model, train_loader, optimizer, scheduler):
    """
    """



def main():
    # Get key arguments
    args = get_args()

    # get the Dataloader for train, validation, and test set
    train_loader, val_loader, test_loader = load_data(args=args)

    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # train

    # load from checkpoint
    model = load_model_from_checkpoint(args, best=True) # after training, get the best model
    
