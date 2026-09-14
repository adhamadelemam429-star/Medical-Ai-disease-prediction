import json
from pathlib import Path

import pandas as pd

from preprocessing.text_preprocessing import combine_fields


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POSSIBLE_INPUTS = [
    PROJECT_ROOT / "medical_chunks_final.json",
    PROJECT_ROOT / "data" / "medical_chunks_final.json",
    PROJECT_ROOT / "notebooks" / "medical_chunks_final.json",
]

OUTPUT_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = OUTPUT_DIR / "training_dataset.csv"


def find_input_file():
    for path in POSSIBLE_INPUTS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find medical_chunks_final.json"
    )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "chunks" in data:
            data = data["chunks"]

        elif "data" in data:
            data = data["data"]

    if not isinstance(data, list):
        raise ValueError(
            "JSON file must contain a list of records."
        )

    return data


def prepare_records(records):
    prepared = []

    for item in records:

        disease = (
            item.get("Disease")
            or item.get("disease")
            or item.get("condition")
            or ""
        )

        section = (
            item.get("Section")
            or item.get("section")
            or ""
        )

        text = (
            item.get("Text")
            or item.get("text")
            or item.get("content")
            or ""
        )

        disease = str(disease).strip()
        section = str(section).strip()
        text = str(text).strip()

        if not disease:
            continue

        if not text:
            continue

        training_text = combine_fields(
            disease=disease,
            section=section,
            text=text
        )

        if not training_text:
            continue

        prepared.append(
            {
                "text": training_text,
                "disease": disease,
                "section": section
            }
        )

    return prepared


def remove_duplicates(records):
    seen = set()
    result = []

    for record in records:

        key = (
            record["disease"].lower(),
            record["text"].lower()
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(record)

    return result


def main():

    print("=" * 60)
    print("PREPARING MEDICAL TRAINING DATASET")
    print("=" * 60)

    input_file = find_input_file()

    print(f"Input file: {input_file}")

    records = load_json(input_file)

    print(
        f"Raw records loaded: {len(records)}"
    )

    records = prepare_records(records)

    print(
        f"Valid records: {len(records)}"
    )

    records = remove_duplicates(records)

    print(
        f"Records after duplicate removal: {len(records)}"
    )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    dataframe = dataframe[
        [
            "text",
            "disease",
            "section"
        ]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(
        f"Samples: {len(dataframe)}"
    )

    print(
        f"Diseases: {dataframe['disease'].nunique()}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()