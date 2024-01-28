from pymongo import MongoClient
import os
from internal.security_service import get_connection_string


def get_collection(db_name: str, collection_name):
    if os.environ.get('ENV')=='local':
        mongo_client = MongoClient('localhost', 27017)
    else:
        mongo_client = MongoClient(get_connection_string())
    return mongo_client[db_name][collection_name]


