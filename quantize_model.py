import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

SOURCE = "models/distilbert_truthlens"
OUTPUT = "models/distilbert_truthlens_int8"

os.makedirs(OUTPUT, exist_ok=True)

print("=" * 70)
print("TruthLens AI - INT8 Model Quantization")
print("=" * 70)

print("Loading original model...")

tokenizer = AutoTokenizer.from_pretrained(SOURCE)

model = AutoModelForSequenceClassification.from_pretrained(
    SOURCE
)

model.eval()

print("Original model loaded.")

print("Applying dynamic INT8 quantization...")

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

quantized_model.eval()

print("Quantization completed.")

print("Saving tokenizer...")

tokenizer.save_pretrained(OUTPUT)

print("Saving quantized model...")

torch.save(
    quantized_model,
    os.path.join(
        OUTPUT,
        "quantized_model.pt",
    ),
)

print("=" * 70)
print("Quantized model saved successfully.")
print("Output:", OUTPUT)
print("=" * 70)