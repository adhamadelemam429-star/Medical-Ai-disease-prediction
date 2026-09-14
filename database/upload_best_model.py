from pathlib import Path

from database.connection import get_database
from gridfs import GridFS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_ROOT / "models" / "active_model.joblib"
VECTORIZER_FILE = PROJECT_ROOT / "models" / "active_vectorizer.joblib"
METADATA_FILE = PROJECT_ROOT / "models" / "best_model_metadata.json"


def upload_file(fs, file_path, file_type):
    print(f"\nUploading: {file_path.name}")

    with open(file_path, "rb") as f:
        file_id = fs.put(
            f,
            filename=file_path.name,
            file_type=file_type,
            project="Medical AI",
        )

    print(f"Uploaded successfully.")
    print(f"GridFS ID: {file_id}")

    return file_id


def main():

    print("=" * 70)
    print("UPLOAD BEST MODEL TO MONGODB GRIDFS")
    print("=" * 70)

    required_files = [
        MODEL_FILE,
        VECTORIZER_FILE,
        METADATA_FILE,
    ]

    print("\nChecking model files...")

    for file_path in required_files:

        print(f"{file_path}")

        if not file_path.exists():
            raise FileNotFoundError(
                f"\nRequired file not found:\n{file_path}"
            )

    db = get_database()

    print(f"\nDatabase: {db.name}")

    fs = GridFS(db)

    model_id = upload_file(
        fs,
        MODEL_FILE,
        "model",
    )

    vectorizer_id = upload_file(
        fs,
        VECTORIZER_FILE,
        "vectorizer",
    )

    metadata_id = upload_file(
        fs,
        METADATA_FILE,
        "metadata",
    )

    print("\n" + "=" * 70)
    print("GRIDFS UPLOAD COMPLETED")
    print("=" * 70)

    print(f"Model ID:      {model_id}")
    print(f"Vectorizer ID: {vectorizer_id}")
    print(f"Metadata ID:   {metadata_id}")

    print("\nGridFS files:")

    for file_info in fs.find({}):
        print(
            f"- {file_info.filename} | "
            f"{file_info.length} bytes"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()