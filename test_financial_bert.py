import os
import torch
import warnings

# Explicitly disable any TensorFlow loading
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

# Ignore warnings
warnings.filterwarnings("ignore")

# Now import the transformers library
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Loading FinancialBERT model and tokenizer from local directory...")

try:
    # Load the tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained("models/financial_bert")

    # Completely removed torch_dtype parameter - no float32 reference at all
    model = AutoModelForMaskedLM.from_pretrained("models/financial_bert")

    print("Successfully loaded model and tokenizer!")
    print(f"Model type: {type(model).__name__}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")

    # Test with a simple example
    text = "The stock market [MASK] today."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    token_id = torch.argmax(
        outputs.logits[0, tokenizer.convert_tokens_to_ids("[MASK]")]
    )
    predicted_token = tokenizer.convert_ids_to_tokens([token_id])[0]

    print(f"\nTest inference:")
    print(f"Input: {text}")
    print(f"Predicted word for [MASK]: {predicted_token}")
    print(f"Result: {text.replace('[MASK]', predicted_token)}")

except Exception as e:
    print(f"An error occurred: {e}")
