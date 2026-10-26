from pathlib import Path

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)


BASE_DIR = Path(__file__).resolve().parent

SOURCE_MODEL = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens_onnx"
    / "model.onnx"
)

OUTPUT_MODEL = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens_onnx"
    / "model.int8.onnx"
)


def main():

    print("=" * 70)
    print("TruthLens AI - ONNX INT8 QUANTIZATION")
    print("=" * 70)

    print("Source model:")
    print(SOURCE_MODEL)

    print("Output model:")
    print(OUTPUT_MODEL)

    if not SOURCE_MODEL.exists():

        raise FileNotFoundError(
            f"ONNX model not found:\n{SOURCE_MODEL}"
        )

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("-" * 70)
    print("Applying dynamic INT8 quantization...")
    print("-" * 70)

    quantize_dynamic(
        model_input=str(SOURCE_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QInt8,
    )

    source_size = SOURCE_MODEL.stat().st_size
    output_size = OUTPUT_MODEL.stat().st_size

    print()
    print(
        f"Original ONNX size: "
        f"{source_size / (1024 * 1024):.2f} MB"
    )

    print(
        f"INT8 ONNX size: "
        f"{output_size / (1024 * 1024):.2f} MB"
    )

    print()
    print("=" * 70)
    print("INT8 ONNX QUANTIZATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()