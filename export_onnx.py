from pathlib import Path

from optimum.exporters.onnx import main_export


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens_onnx"
)


def main():

    print("=" * 70)
    print("TruthLens AI - ONNX EXPORT")
    print("=" * 70)

    print("Source model:")
    print(MODEL_DIR)

    print("Output directory:")
    print(OUTPUT_DIR)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}"
        )

    config_file = MODEL_DIR / "config.json"
    weights_file = MODEL_DIR / "model.safetensors"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Missing config.json: {config_file}"
        )

    if not weights_file.exists():
        raise FileNotFoundError(
            f"Missing model.safetensors: {weights_file}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("-" * 70)
    print("Exporting DistilBERT to ONNX...")
    print("-" * 70)

    main_export(
        model_name_or_path=str(MODEL_DIR),
        output=str(OUTPUT_DIR),
        task="text-classification",
        framework="pt",
        opset=17,
        device="cpu",
        local_files_only=True,
        do_validation=True,
    )

    print("-" * 70)
    print("Copying tokenizer files...")
    print("-" * 70)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]

    for filename in tokenizer_files:

        source = MODEL_DIR / filename
        destination = OUTPUT_DIR / filename

        if source.exists():

            destination.write_bytes(
                source.read_bytes()
            )

            print(
                "Copied:",
                filename,
            )

    print("=" * 70)
    print("ONNX EXPORT COMPLETED")
    print("=" * 70)

    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()