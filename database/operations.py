from database.connection import get_database

from database.config import (
    CONDITIONS_COLLECTION
)


def get_conditions_collection():

    database = get_database()

    return database[
        CONDITIONS_COLLECTION
    ]


def get_condition(
    condition_name
):

    collection = (
        get_conditions_collection()
    )

    return collection.find_one(
        {
            "condition": condition_name
        }
    )


def get_all_conditions():

    collection = (
        get_conditions_collection()
    )

    return list(
        collection.find(
            {},
            {
                "_id": 0
            }
        )
    )


def save_condition(
    condition
):

    collection = (
        get_conditions_collection()
    )

    return collection.update_one(
        {
            "condition":
                condition["condition"]
        },
        {
            "$set": condition
        },
        upsert=True
    )