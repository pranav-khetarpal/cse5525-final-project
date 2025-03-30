import os
import sys
import warnings

# Explicitly disable any TensorFlow loading
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

# Ignore warnings
warnings.filterwarnings("ignore")

print("Loading libraries...")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    print("Loading FinancialBERT model and tokenizer from local directory...")

    # Use relative path with ./ to ensure correct directory is found
    model_path = "./financial_bert_model"

    # First load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully!")

    # Try loading the model with various compatibility options
    try:
        # Try first without any dtype specification
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    except Exception as e1:
        print(f"First loading attempt failed: {e1}")
        try:
            # For older PyTorch versions
            model = AutoModelForMaskedLM.from_pretrained(
                model_path, torch_dtype=torch.float
            )
        except Exception as e2:
            print(f"Second loading attempt failed: {e2}")
            try:
                # For newer PyTorch versions with low_cpu_mem_usage flag
                model = AutoModelForMaskedLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True
                )
            except Exception as e3:
                print(f"Third loading attempt failed: {e3}")
                print(
                    "All loading attempts failed. Try using a different PyTorch/transformers version."
                )
                sys.exit(1)

    print("Successfully loaded model and tokenizer!")
    print(f"Model type: {type(model).__name__}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")

    # Test with a simple example
    text = "The stock market [MASK] today."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Find the position of the [MASK] token
    mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    # Get the predicted token
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_token_id = torch.argmax(mask_token_logits, dim=-1)
    predicted_token = tokenizer.decode(top_token_id)

    print(f"\nTest inference:")
    print(f"Input: {text}")
    print(f"Predicted word for [MASK]: {predicted_token}")
    print(f"Result: {text.replace('[MASK]', predicted_token)}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting tips:")
    print(
        "1. Make sure you're in the correct directory (should be in the FinancialBert folder)"
    )
    print("2. Check that the model files exist in ./models/financial_bert/")
    print(
        "3. Try installing compatible versions: pip install torch==2.2.0 transformers==4.30.0"
    )
    print(
        "4. If on Apple Silicon Mac, use Python 3.9+ with appropriate PyTorch version"
    )
