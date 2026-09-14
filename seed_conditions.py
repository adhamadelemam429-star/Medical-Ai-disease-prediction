import json
from pathlib import Path

from database.connection import get_database


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "notebooks" / "medical_diseases_cleaned.json"

COLLECTION_NAME = "medical_chunks"


def load_diseases():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def extract_text(record):
    if isinstance(record, str):
        return record

    if not isinstance(record, dict):
        return ""

    parts = []

    for key, value in record.items():
        if value is None:
            continue

        if isinstance(value, list):
            value = " ".join(str(x) for x in value)

        elif isinstance(value, dict):
            value = json.dumps(
                value,
                ensure_ascii=False
            )

        else:
            value = str(value)

        parts.append(f"{key}: {value}")

    return "\n".join(parts)


def main():
    print("=" * 70)
    print("SYNC MEDICAL DATA TO MONGODB")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATA_FILE)

    diseases = load_diseases()

    print(f"\nLoaded records: {len(diseases)}")

    db = get_database()

    collection = db[COLLECTION_NAME]

    print(f"Database: {db.name}")
    print(f"Collection: {COLLECTION_NAME}")

    inserted = 0
    updated = 0

    for index, record in enumerate(diseases, start=1):

        if not isinstance(record, dict):
            continue

        disease_name = (
            record.get("disease")
            or record.get("name")
            or record.get("title")
            or f"unknown_{index}"
        )

        text = extract_text(record)

        document = {
            "disease": disease_name,
            "text": text,
            "source": "NHS Inform",
            "record_index": index,
            "raw_data": record,
        }

        result = collection.update_one(
            {
                "disease": disease_name,
                "record_index": index,
            },
            {
                "$set": document
            },
            upsert=True,
        )

        if result.upserted_id is not None:
            inserted += 1
        else:
            updated += 1

        if index % 50 == 0 or index == len(diseases):
            print(
                f"Processed: {index}/{len(diseases)}"
            )

    print("\n" + "=" * 70)
    print("SYNC COMPLETED")
    print("=" * 70)

    print(f"Inserted: {inserted}")
    print(f"Updated: {updated}")

    print(
        f"MongoDB collection documents: "
        f"{collection.count_documents({})}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()