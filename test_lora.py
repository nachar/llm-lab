import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = "/Users/hassennachar/developments/2026/ia-practice/models/llama-3.2-1b"
adapter_path = "./lora-adapter"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_path)

model = AutoModelForCausalLM.from_pretrained(
    base_path,
    dtype=torch.float16,
).to(device)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

prompt = "### Pregunta:\n¿Qué diferencia hay entre GET y POST?\n### Respuesta:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
