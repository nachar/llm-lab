import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/Users/hassennachar/developments/2026/ia-practice/models/llama-3.2-1b"

# Comprobar aceleración disponible
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Dispositivo: {device}")

# Cargar tokenizer
print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Cargar modelo en FP16
print("Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16
).to(device)

print(f"Modelo cargado: {model.config.model_type}")
print(f"Parámetros: {model.num_parameters():,}")
print(f"Precisión: {model.dtype}")

# Prompt de prueba
prompt = "### Pregunta:\n¿Qué diferencia hay entre GET y POST?\n### Respuesta:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generar respuesta
print("\nGenerando respuesta...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\nRespuesta del modelo:")
print(response)