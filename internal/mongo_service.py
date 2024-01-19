from pymongo import MongoClient

from config_anachron import settings

__mongo_client__ = None


# connects to a mongo db server (if there isn't a connection yet)
# based on working environment variable env
# returns a mongo connection
def get_connection():
    global __mongo_client__
    if __mongo_client__ is not None:
        return __mongo_client__
    if settings.env == "local":
        __mongo_client__ = MongoClient('localhost', 27017)
    elif settings.env == "qa" or settings.env == "prod":
        uri = settings.db_connection_string
        __mongo_client__ = MongoClient(uri)
    else:
        print(settings.env)
        raise Exception("other environments are not ready yet")
    return __mongo_client__


# db_name is the client_id for now
def get_db(db_name: str):
    return get_connection()[db_name]


def get_collection(db_name: str, collection_name):
    return get_connection()[db_name][collection_name]

