from pymongo import MongoClient
import os


def get_collection(db_name: str, collection_name):
    if os.environ.get('ENV')=='local':
        mongo_client = MongoClient('localhost', 27017)
    else:
        mongo_client = MongoClient(os.environ.get('DB_CONNECTION_STRING'))
    return mongo_client[db_name][collection_name]


