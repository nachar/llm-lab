
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/Users/hassennachar/developments/2026/ia-practice/models/finance-specialist-v7"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Dispositivo: {device}")

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16
).to(device)

model.eval()

print(f"Modelo cargado: {model.config.model_type}")
print(f"Parámetros: {model.num_parameters():,}")
print(f"Precisión: {model.dtype}")

messages = [
    {
        "role": "system",
        "content": "You are a financial analyst specialized in stocks and investment analysis. Provide clear, objective and evidence-based financial analysis. Never invent data."
    },
    {
        "role": "user",
        "content": """
Analyze this scenario:

Apple stock has dropped 5% in one day.
The company reports strong quarterly earnings, but announces lower revenue guidance for next quarter.

Explain:
1. What might be driving the price movement?
2. What are the bullish and bearish signals?
3. What risks should an investor consider?
4. What additional information would you need?

Provide a concise, structured analysis.
"""
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerando respuesta...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nRespuesta del modelo:")
print(response)
