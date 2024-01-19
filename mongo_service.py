from pymongo import MongoClient


def get_collection(db_name: str, collection_name):
    mongo_client = MongoClient('localhost', 27017)
    return mongo_client[db_name][collection_name]

