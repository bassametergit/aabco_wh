from jose import jwt
from passlib.context import CryptContext
import json
import os
PRIVATE_KEY = open("PrivateKey.txt").read()
PUBLIC_KEY = open("PublicKey.txt").read()

ALGORITHM = "RS256"

__pwd_context__ = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str, context=__pwd_context__) -> str:
    return context.hash(password)


def verify_password(plain_password: str, hashed_password: str, context=__pwd_context__) -> bool:
    return context.verify(plain_password, hashed_password)


def decode_token(token: str, public_key=PUBLIC_KEY, algorithm=ALGORITHM):
    return jwt.decode(token, public_key, algorithms=[algorithm])


def create_access_token(data: dict, private_key=PRIVATE_KEY, algorithm=ALGORITHM):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, private_key, algorithm=algorithm)
    return encoded_jwt

def get_openai_and_pinecone_keys():
    if os.environ.get('ENV')=='local':
        return os.environ.get('OPENAI_API_KEY'), os.environ.get('PINECONE_API_KEY'), os.environ.get('PINECONE_ENVIRONMENT')
    else:
        with open('config.json') as f:
            config = json.load(f)
            return config.get('openai_secret_key'), config.get('pinecone_secret_key'), config.get('pinecone_environment')
        
def get_connection_string():
    if os.environ.get('ENV')=='local':
        return os.environ.get('DB_CONNECTION_STRING')
    else:
        with open('config.json') as f:
            config = json.load(f)
            return config.get('connection_string')