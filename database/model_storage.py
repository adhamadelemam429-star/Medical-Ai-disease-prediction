from datetime import datetime, timezone
from pathlib import Path

from gridfs import GridFS

from database.connection import get_database


def save_model_to_gridfs(
    model_path,
    model_name,
    model_type,
    metrics=None
):

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    database = get_database()

    gridfs = GridFS(
        database
    )

    with open(
        model_path,
        "rb"
    ) as file:

        file_id = gridfs.put(
            file,
            filename=model_path.name,
            model_name=model_name,
            model_type=model_type,
            upload_date=datetime.now(
                timezone.utc
            )
        )

    database.models.update_one(
        {
            "model_name": model_name
        },
        {
            "$set": {
                "model_name": model_name,
                "model_type": model_type,
                "gridfs_id": file_id,
                "metrics": metrics or {},
                "filename": model_path.name,
                "updated_at": datetime.now(
                    timezone.utc
                )
            }
        },
        upsert=True
    )

    return file_id


def list_stored_models():

    database = get_database()

    return list(
        database.models.find(
            {},
            {
                "_id": 0
            }
        )
    )