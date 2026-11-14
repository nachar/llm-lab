from datasets import load_dataset
from transformers import AutoTokenizer

model_path = "/Users/hassennachar/developments/2026/ia-practice/models/llama-3.2-1b"

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

print(tokenized_dataset)
print("Primer ejemplo tokenizado:")
print(tokenized_dataset[0]["input_ids"])
