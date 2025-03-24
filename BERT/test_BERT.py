import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# function to preprocess the tweet and get the tokenized inputs
def tokenize_tweet(text, tokenizer, max_length=128):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


# load CSV file
test_file = "../tweet/processed_test_stockemo.csv"
df = pd.read_csv(test_file)

# get the cleaned tweet column and the sentiment labels
tweets = df["cleaned_tweet"].tolist()
true_labels_str = df["senti_label"].tolist() # original string labels

# map string labels to numerical values
label_mapping = {"bearish": 0, "bullish": 1}
true_labels = [label_mapping[label] for label in true_labels_str]  # Convert to numeric

# load BERT model and tokenizer from the 'bert_model' folder
save_directory = "./bert_model"
tokenizer = BertTokenizer.from_pretrained(save_directory)
model = BertForSequenceClassification.from_pretrained(save_directory, num_labels=2)  # 2 labels for binary classification

# Set model to evaluation mode
model.eval()

# initialize a list to store predictions
predictions = []

# iterate over each tweet and perform inference
for tweet in tweets:
    # preprocess the tweet to get tokenized input
    inputs = tokenize_tweet(tweet, tokenizer)

    # perform inference (without calculating gradients)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # convert logits to class probabilities (softmax)
    probabilities = nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # append prediction to the list
    predictions.append(predicted_class)

# compute evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average="binary")
recall = recall_score(true_labels, predictions, average="binary")
f1 = f1_score(true_labels, predictions, average="binary")

# output the results
num_examples = len(true_labels)
print(f"Number of examples tested: {num_examples}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
