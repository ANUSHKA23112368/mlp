import csv
import json
import sys

def csv_to_json(csv_path: str, json_path: str):
    """Utility to convert tabular complaint datasets to structured JSON lines."""
    with open(csv_path, mode="r", encoding="utf-8") as f_in, open(json_path, mode="w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            f_out.write(json.dumps(row) + "\n")
    print(f"Exported JSON Lines to {json_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        csv_to_json(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python export_jsonl.py <input.csv> <output.jsonl>")
