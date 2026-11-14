import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

model_path = "/Users/hassennachar/developments/2026/ia-practice/models/llama-3.2-1b"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
).to(device)

model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    use_cpu=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)

trainer.train()
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")

print("Adaptador LoRA guardado en ./lora-adapter")
