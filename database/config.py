import os


MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/"
)

DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "medical_ai"
)

CONDITIONS_COLLECTION = os.getenv(
    "CONDITIONS_COLLECTION",
    "conditions"
)

MODELS_COLLECTION = os.getenv(
    "MODELS_COLLECTION",
    "models"
)