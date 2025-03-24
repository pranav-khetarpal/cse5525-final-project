import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

print("Loading the BERT model and tokenizer...")

# Use absolute path to the bert_model folder
save_directory = (
    "/Users/adewaleadenle/Downloads/Dev/Junior/cse5525-final-project/BERT/bert_model"
)
tokenizer = BertTokenizer.from_pretrained(save_directory)
model = BertModel.from_pretrained(save_directory)

# Check if it's a sequence classification model or a base model
if isinstance(model, BertModel):
    print("Loaded a base BERT model, not a classification model.")
    print(
        "This might not be suitable for direct sentiment classification without additional layers."
    )

    # Option to convert to classification model if needed:
    # num_labels = 3  # bullish, bearish, neutral
    # classification_model = BertForSequenceClassification.from_pretrained(
    #     save_directory, num_labels=num_labels, from_tf=False
    # )
else:
    print("Loaded a classification-ready BERT model.")

print("Model and tokenizer loaded successfully!")

# Define the test file path
test_file = "tweet/processed_test_stockemo.csv"

# Load the test data
print(f"Loading test data from {test_file}...")
test_df = pd.read_csv(test_file)
print(f"Loaded {len(test_df)} test examples")

# Display the first few rows
print("\nSample test data:")
print(test_df.head())

# Define sentiment mapping (bullish -> 2, bearish -> 0)
sentiment_map = {
    "bullish": 2,
    "bearish": 0,
}
test_df["label"] = test_df["senti_label"].map(sentiment_map)


# If you have a base model, you need a classification head
# This is a simple example - you might need to adjust based on your model:
class SentimentClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_labels=3):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Setup classifier if needed
if isinstance(model, BertModel):
    print("Creating a simple sentiment classifier using the base BERT model...")
    classifier = SentimentClassifier(model, num_labels=3)
    # You would need to load your trained weights here
    # classifier.load_state_dict(torch.load('path_to_classifier_weights.pt'))
    classifier.eval()
else:
    classifier = model
    classifier.eval()


# Define a function to make predictions
def predict_sentiment(texts, model, tokenizer, max_length=128, is_base_model=True):
    predictions = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        with torch.no_grad():
            if is_base_model:
                # For base model with separate classifier
                logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                # Since we don't have trained weights for the classifier,
                # we'll do a simple heuristic prediction based on sentiment keywords
                # This is just for demonstration and not very accurate
                text_lower = text.lower()
                if any(
                    word in text_lower
                    for word in [
                        "good",
                        "great",
                        "up",
                        "rise",
                        "high",
                        "bullish",
                        "buy",
                        "profit",
                    ]
                ):
                    predictions.append(2)  # bullish
                elif any(
                    word in text_lower
                    for word in [
                        "bad",
                        "poor",
                        "down",
                        "fall",
                        "low",
                        "bearish",
                        "sell",
                        "loss",
                    ]
                ):
                    predictions.append(0)  # bearish
                else:
                    predictions.append(1)  # neutral
            else:
                # For sequence classification model
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                predictions.append(predicted_class)

    return predictions


print("\nNote: Since we have a base BERT model without a trained classification head,")
print("we'll use a simple keyword-based heuristic for demonstration purposes.")
print("For proper sentiment classification, you would need:")
print("1. A fine-tuned BERT classification model, or")
print("2. A classification head trained on top of this base model")

# Make predictions on test data
print("\nMaking predictions on test data...")
is_base_model = isinstance(model, BertModel)
test_predictions = predict_sentiment(
    test_df["cleaned_tweet"].tolist(),
    classifier,
    tokenizer,
    is_base_model=is_base_model,
)

# Evaluate the model
accuracy = accuracy_score(test_df["label"], test_predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification report with appropriate class names
target_names = ["bearish", "neutral", "bullish"]
print("\nClassification Report:")
print(
    classification_report(test_df["label"], test_predictions, target_names=target_names)
)

print("\nConfusion Matrix:")
print(confusion_matrix(test_df["label"], test_predictions))

# Calculate number of predictions in each class
class_counts = {
    "bearish (0)": test_predictions.count(0),
    "neutral (1)": test_predictions.count(1),
    "bullish (2)": test_predictions.count(2),
}
print("\nPrediction class distribution:")
for label, count in class_counts.items():
    print(f"{label}: {count} ({count/len(test_predictions)*100:.1f}%)")

print(
    "\nNote: These results are based on a heuristic approach and not a trained model."
)
print("For better results, consider fine-tuning the BERT model on your specific task.")
