import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn as nn

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
true_labels = df["senti_label"].tolist()


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

    # map the predicted class to the sentiment (0 -> bearish, 1 -> bullish)
    sentiment_map = {0: "bearish", 1: "bullish"}
    predicted_sentiment = sentiment_map[predicted_class]

    # append prediction to the list
    predictions.append(predicted_sentiment)

# add the predictions to the DataFrame for evaluation
df['predicted_sentiment'] = predictions


# compare the predictions with the true labels
df['correct'] = df['predicted_sentiment'] == df['senti_label']
accuracy = df['correct'].mean()
print(f"Accuracy: {accuracy:.4f}")
