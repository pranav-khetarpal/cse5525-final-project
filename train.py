import argparse
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"], help="What optimizer to use")
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
        input_tensors = tokenizer(data_frame['cleaned_tweet'], return_tensors="pt") # TODO this is maybe a dictionary, not actual Tensors

        dataset = TensorDataset(tensors=input_tensors) # create TensorDatasets to pass to DataLoaders
        return dataset

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH) # TODO tokenizer should not matter whether we are training BERT or FinancialBERT

    train_dataset = process_file(tokenizer=tokenizer, file_path=TRAIN_DATA) # getting TensorDatasets, which is basically like Dataset
    val_dataset = process_file(tokenizer=tokenizer, file_path=VAL_DATA)
    test_dataset = process_file(tokenizer=tokenizer, file_path=TEST_DATA)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True) # getting DataLoaders, only shuffle in training
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader)


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


def main():
    # Get key arguments
    args = get_args()

    # get the Dataloader for train, validation, and test set
    train_loader, val_loader, test_loader = load_data(args=args)

    model = initialize_model(args)
    # optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))




    