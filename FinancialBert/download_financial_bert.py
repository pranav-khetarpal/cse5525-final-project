import os
from huggingface_hub import snapshot_download

# Create directory if it doesn't exist
os.makedirs("financial_bert_model", exist_ok=True)

print("Downloading FinancialBERT model and tokenizer...")

try:
    # Download the model using huggingface_hub
    # This is just for downloading the model files to disk without loading them into memory, 
    # This approach avoids TensorFlow dependency issues during download
    model_path = snapshot_download(
        repo_id="ahmedrachid/FinancialBERT",
        local_dir="financial_bert_model",
        local_dir_use_symlinks=False,
    )

    print(f"FinancialBERT model and tokenizer downloaded successfully to: {model_path}")

except Exception as e:
    print(f"An error occurred: {e}")
