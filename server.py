
import json
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/Users/hassennachar/developments/2026/ia-practice/models/finance-specialist-v7"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16
).to(device)

model.eval()

print("Finance Specialist ready!")

app = FastAPI(title="Market Mind AI")


class AnalysisRequest(BaseModel):
    market_data: dict
    news: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok", "model": "finance-specialist-v7"}


@app.post("/analyze")
def analyze(request: AnalysisRequest):

    context = f"""
    Analyze the following financial information.

    MARKET DATA:
    {json.dumps(request.market_data, indent=2)}

    FINANCIAL NEWS:
    {json.dumps(request.news, indent=2)}

    Identify important movements, bullish and bearish signals,
    relevant news, risks and potential opportunities.

    Do not invent information. Explain your reasoning.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst specialized in stocks, ETFs and cryptocurrencies."
        },
        {
            "role": "user",
            "content": context
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return {"analysis": response}
