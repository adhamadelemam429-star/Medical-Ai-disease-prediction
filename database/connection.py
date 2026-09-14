from pymongo import MongoClient

from database.config import (
    MONGO_URI,
    DATABASE_NAME
)


_client = None


def get_client():

    global _client

    if _client is None:

        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000
        )

    return _client


def get_database():

    client = get_client()

    return client[
        DATABASE_NAME
    ]


def test_connection():

    client = get_client()

    client.admin.command(
        "ping"
    )

    return True