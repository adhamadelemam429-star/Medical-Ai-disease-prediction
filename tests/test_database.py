from database.connection import test_connection as check_database_connection


def test_mongodb_connection():
    assert check_database_connection() is True