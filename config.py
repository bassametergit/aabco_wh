from pydantic import BaseSettings


class Settings(BaseSettings):
    env: str
    openai_api_key: str
    pinecone_api_key: str
    index_name: str
    db_connection_string: str    
    db_name: str


def get_settings():
    return Settings()

settings = get_settings()