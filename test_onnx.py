from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens_onnx"
)

MODEL_FILE = MODEL_DIR / "model.int8.onnx"


def main():

    print("=" * 70)
    print("TruthLens AI - INT8 ONNX TEST")
    print("=" * 70)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"INT8 ONNX model not found:\n{MODEL_FILE}"
        )

    print("Model:")
    print(MODEL_FILE)

    print("-" * 70)
    print("Loading tokenizer...")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    print("Tokenizer loaded.")

    print("-" * 70)
    print("Loading ONNX Runtime model...")
    print("-" * 70)

    session = ort.InferenceSession(
        str(MODEL_FILE),
        providers=["CPUExecutionProvider"],
    )

    print("ONNX model loaded.")

    print()
    print("Available inputs:")

    for item in session.get_inputs():
        print(
            " ",
            item.name,
            item.shape,
            item.type,
        )

    print()
    print("Available outputs:")

    for item in session.get_outputs():
        print(
            " ",
            item.name,
            item.shape,
            item.type,
        )

    text = "The Earth is flat"

    print("-" * 70)
    print("Testing claim:")
    print(text)
    print("-" * 70)

    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
    )

    inputs = {}

    available_inputs = {
        item.name
        for item in session.get_inputs()
    }

    for name in available_inputs:

        if name in encoded:

            inputs[name] = encoded[name].astype(
                np.int64
            )

    print("Inputs supplied:")

    for name, value in inputs.items():
        print(
            " ",
            name,
            value.shape,
            value.dtype,
        )

    outputs = session.run(
        None,
        inputs,
    )

    print()
    print("Raw outputs:")

    for output in outputs:
        print(
            " ",
            output.shape,
            output.dtype,
        )

    logits = outputs[0][0]

    print()
    print("Logits:")
    print(logits)

    # Numerically stable softmax
    shifted = logits - np.max(logits)

    probabilities = np.exp(shifted)

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    print()
    print("Probabilities:")

    for index, probability in enumerate(
        probabilities
    ):
        print(
            f"  Class {index}: "
            f"{probability * 100:.2f}%"
        )

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class] * 100
    )

    print()
    print("Predicted class:")
    print(predicted_class)

    print(
        f"Model confidence: {confidence:.2f}%"
    )

    print("=" * 70)
    print("INT8 ONNX TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()