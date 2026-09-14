# =========================================================
# Medical AI - Configuration
# =========================================================

MONGO_URI = "mongodb://localhost:27017"

DATABASE_NAME = "medical_ai"

COLLECTION_NAME = "medical_chunks"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

LLM_MODEL_NAME = "qwen3:8b"

TOP_K = 5

MIN_SIMILARITY = 0.30

MAX_CONTEXT_CHARS = 8000