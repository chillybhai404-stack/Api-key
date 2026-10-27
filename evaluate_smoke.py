"""Run the included out-of-training smoke-test phrases.

This is a sanity check, not a scientific benchmark or a guarantee of real-world accuracy.
"""
from __future__ import annotations
import csv
from pathlib import Path
from model_utils import predict_text

ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "data" / "smoke_tests.csv"

def main():
    total = primary_hits = full_hits = 0
    with TEST_PATH.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected = set(row["expected_labels"].split("|"))
            result = predict_text(row["text"])
            got = {item["label"] for item in result["labels"]}
            primary = bool(expected & got)
            full = expected.issubset(got)
            total += 1
            primary_hits += int(primary)
            full_hits += int(full)
            mark = "PASS" if primary else "MISS"
            print(f"{mark} | expected={','.join(sorted(expected))} | got={','.join(sorted(got))}\n  {row['text']}")
    print("\nSmoke-test summary")
    print(f"At least one expected label returned: {primary_hits}/{total}")
    print(f"All expected labels returned: {full_hits}/{total}")
    print("These examples are a basic sanity check, not a valid estimate of production accuracy.")

if __name__ == "__main__":
    main()
