import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

# Explicitly disable any TensorFlow loading
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

# Ignore warnings
warnings.filterwarnings("ignore")


print("Loading libraries...")
output_file = "results.txt"

# This function writes the results to a txt file
def write_results_to_file(results, file_path=output_file):
    with open(file_path, "a") as file:
        file.write(results)


try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    # Define the test file path
    test_file = "../tweet/processed_test_stockemo.csv"

    # Load the test data
    print(f"Loading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(test_df)} test examples")

    # Display the first few rows
    print("\nSample test data:")
    print(test_df.head())

    # Map sentiment labels to numerical values
    sentiment_map = {"bullish": 1, "bearish": 0}
    test_df["label"] = test_df["senti_label"].map(sentiment_map)

    print("\nLoading FinancialBERT model and tokenizer .")

    model_path = "financial_bert_model"
    # Load tokenizer and model directly from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully!")

    # Load the model for sequence classification
    try:
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        print(f"Successfully loaded sequence classification model from {model_path}!")
    except Exception as e1:
        print(
            f"Loading sequence classification model from local directory failed: {e1}"
        )
        print("Attempting to load model from HuggingFace instead...")

    # Prepare for evaluation
    model.eval()

    # Define a function to make predictions
    def predict_sentiment(texts, model, tokenizer, max_length=128):
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
                outputs = model(**inputs)

            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class)

        return predictions

    # Make predictions on test data
    print("\nMaking predictions on test data...")
    test_predictions = predict_sentiment(
        test_df["cleaned_tweet"].tolist(), model, tokenizer
    )

    # Evaluate the model
    accuracy = accuracy_score(test_df["label"], test_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    write_results_to_file(
        "Test Accuracy: " + str(accuracy)
    )
    print("\nClassification Report:")
    print(
        classification_report(
            test_df["label"], test_predictions, target_names=["bearish", "bullish"]
        )
    )
    write_results_to_file(
        "Classification Report: " + str(classification_report(test_df["label"], test_predictions, target_names=["bearish", "bullish"]))
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_df["label"], test_predictions))
    write_results_to_file(
        "Confusion Matrix: " + str(confusion_matrix(test_df["label"], test_predictions))
    )

except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting tips:")
    print(
        "1. Make sure you're in the correct directory (should be in the FinancialBert folder)"
    )
    print("2. Check that the model files exist in ./models/financial_bert/")
    print("3. Make sure the test file exists and has the expected format")
    print(
        "4. Try installing compatible versions: pip install torch==2.2.0 transformers==4.30.0 pandas scikit-learn"
    )
    print(
        "5. If on Apple Silicon Mac, use Python 3.9+ with appropriate PyTorch version"
    )
