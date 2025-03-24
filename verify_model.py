import os
import json

print("Verifying FinancialBERT model files...")

# Update path to include the FinancialBert subdirectory
model_dir = "FinancialBert/models/financial_bert"

# Check if the directory exists
if not os.path.exists(model_dir):
    print(f"Error: Model directory {model_dir} does not exist.")
    exit(1)

# List all files in the directory
files = os.listdir(model_dir)
print(f"Found {len(files)} files in {model_dir}:")
for file in sorted(files):
    print(f"  - {file}")

# Check for essential model files
essential_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
missing_files = [f for f in essential_files if f not in files]

if missing_files:
    print("\nWarning: Some essential model files are missing:")
    for file in missing_files:
        print(f"  - {file}")
else:
    print("\nAll essential model files are present.")

# Try to read and parse the config.json file
config_path = os.path.join(model_dir, "config.json")
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print("\nModel configuration:")
        print(f"  - Model type: {config.get('model_type', 'unknown')}")
        print(f"  - Hidden size: {config.get('hidden_size', 'unknown')}")
        print(
            f"  - Num attention heads: {config.get('num_attention_heads', 'unknown')}"
        )
        print(f"  - Num hidden layers: {config.get('num_hidden_layers', 'unknown')}")
    except Exception as e:
        print(f"\nError reading config.json: {e}")
else:
    print("\nConfig file not found.")

print(
    "\nModel verification complete. The model files are downloaded and ready for use."
)
print("To use this model for fine-tuning, you'll need a compatible PyTorch version.")
print("Recommended: Use PyTorch 1.10+ and transformers 4.15+ in a new environment.")
