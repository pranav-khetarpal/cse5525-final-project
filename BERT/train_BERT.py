from transformers import BertTokenizer, BertModel

# Load from the bert_model folder
save_directory = "./bert_model"
tokenizer = BertTokenizer.from_pretrained(save_directory)
model = BertModel.from_pretrained(save_directory)

print("Model and tokenizer loaded successfully!")
