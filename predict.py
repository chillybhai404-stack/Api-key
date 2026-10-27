"""Terminal chat for testing AXU Interaction Intelligence v2."""
from __future__ import annotations
import json
from model_utils import predict_text, load_bundle


def main():
    bundle = load_bundle()
    print(f"{bundle.get('name', 'AXU Interaction Intelligence')} v{bundle.get('version', '2.0')}")
    print("Enter a message to classify. Type 'quit' to stop.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        try:
            print(json.dumps(predict_text(text), ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"Prediction error: {exc}")

if __name__ == "__main__":
    main()
