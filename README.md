# Financial Sentiment Analysis with BERT and FinancialBERT

This project fine-tunes BERT and FinancialBERT models for financial sentiment analysis on Twitter data, classifying tweets as either bullish (positive) or bearish (negative).

## Project Overview

Financial sentiment analysis is crucial for understanding market sentiment from social media. This project:

1. Fine-tunes pre-trained BERT and domain-specific FinancialBERT models
2. Evaluates performance on financial tweet sentiment classification
3. Provides tools for processing and analyzing financial Twitter data

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- PyTorch
- Transformers
- pandas
- scikit-learn
- tqdm

### Model Download

Before training, download the pre-trained models:

For BERT:

```bash
mkdir -p BERT/bert_model
# Download BERT base-uncased model
python -c "from transformers import BertTokenizer, BertForSequenceClassification; BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained('BERT/bert_model'); BertForSequenceClassification.from_pretrained('bert-base-uncased').save_pretrained('BERT/bert_model')"
```

For FinancialBERT:

```bash
python FinancialBert/download_financial_bert.py
```

## Data Preparation

The project uses:

- StockEmo dataset: Stock-related tweets with sentiment labels
- ZeroShot dataset: `zeroshot/twitter-financial-news-sentiment` from HuggingFace

Process the data with:

```bash
# Process StockEmo data
python utils/stockemo_tweet_processing.py

# Process ZeroShot data
python utils/validation_zeroshot_tweet_preprocessing.py

# Merge datasets (if needed)
python utils/combine_zeroshot_with_stockemo.py
```

## Training Models

Train BERT or FinancialBERT with different configurations:

### Basic Training

```bash
python train.py \
  --model_name BERT \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --max_n_epochs 10 \
  --patience_epochs 3 \
  --batch_size 16 \
  --experiment_name bert_base_experiment
```

### Train FinancialBERT

```bash
python train.py \
  --model_name FinancialBERT \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --scheduler_type linear \
  --num_warmup_epochs 1 \
  --max_n_epochs 8 \
  --patience_epochs 2 \
  --batch_size 16 \
  --experiment_name financial_bert_experiment
```

## Evaluation

The training script automatically evaluates the model on validation and test data after training. Results include:

- Loss
- F1 Score

For separate evaluation of a trained model:

```bash
python evaluate.py \
  --model_name BERT \
  --experiment_name bert_base_experiment \
  --batch_size 32
```

## Testing Individual Tweets

Test a trained model on custom tweets:

```bash
python predict.py \
  --model_name FinancialBERT \
  --experiment_name financial_bert_experiment \
  --input "The stock market is showing signs of recovery after the recent dip."
```

## Project Structure

```
cse5525-final-project/
├── BERT/                      # BERT model and testing
├── FinancialBert/             # FinancialBERT model and testing
├── checkpoint/                # Saved model checkpoints
├── tweet/                     # Processed tweet datasets
├── utils/                     # Utility scripts for data processing
├── train.py                   # Main training script
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation
```

## Results

Performance metrics from our experiments:

| Model         | Accuracy | F1 Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| BERT          | 0.44     | 0.13     | 0.49      | 0.08   |
| FinancialBERT | 0.51     | 0.51     | 0.51      | 0.51   |

FinancialBERT consistently outperforms standard BERT on financial sentiment tasks, demonstrating the value of domain-specific pre-training.

## Data Sources

- StockEmo dataset: Synthetic stock market tweets with sentiment labels
- ZeroShot dataset: Available through Hugging Face

```python
from datasets import load_dataset
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
```

## Acknowledgments

- FinancialBERT: https://huggingface.co/ahmedrachid/FinancialBERT
- BERT: https://huggingface.co/bert-base-uncased
- HuggingFace Transformers: https://github.com/huggingface/transformers
