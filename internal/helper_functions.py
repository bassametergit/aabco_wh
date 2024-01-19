import datetime
import json
from bson import ObjectId


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId) or isinstance(o, datetime.datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)


def jsonify_output(input):
    return json.loads(CustomJSONEncoder().encode(input))