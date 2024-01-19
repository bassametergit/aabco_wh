from typing import List, Optional, Union, Dict, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from langchain.docstore.document import Document
from langchain.chains.query_constructor.base import AttributeInfo
from typing import Any

class UserForApi(BaseModel):
    userName: str = Field("User", description="User Name")
    userFrontendId:str= Field(..., description="User Frontend Id")
    email:str= Field(..., description="User Email")
    role:str= Field(..., description="User Role. One of (superadmin, admin, creator, user)")
    def serialize_input(self):
        return {
            'userName': self.userName,
            'userFrontendId': self.userFrontendId,
            'email': self.email,
            'role': self.role
        }
 
class DataFoldersForGetNamespaces(BaseModel):
    id:str
    name: str 
    creationDate: datetime 
    description:str

    def __init__(self, id:str,  name: str, creationDate: datetime , description:str):
        super().__init__(
            id=id,
            name=name,
            creationDate=creationDate,
            description=description
        )
   
        
class SessionForUserSessions(BaseModel):
    sessionId:str
    title: str 
    startDate: datetime 
    chatHistory: List[List[str]]
    lastChatHistoryUpdate:datetime


    def __init__(self, sessionId:str, title: str, startDate: datetime, 
                 chatHistory:List[List[str]], lastChatHistoryUpdate:datetime):
        super().__init__(
            sessionId=sessionId,
            title=title,
            startDate=startDate,
            chatHistory=chatHistory,
            lastChatHistoryUpdate=lastChatHistoryUpdate,
        )
        

class CustomChatbotResponse(BaseModel):
    """Chat response schema."""

    sender: str
    type: str
    answer: str
    source_documents: Optional[List[Document]] = []
    updated_chat_history: Optional[List[List[str]]] = []
    


class DataFolderForApi(BaseModel):
    nsName: str = Field("", description="A name for the Data Folder")
    indexName: str = Field("", description="Index name in which this namespace should be created  in Pinecone")
    docs:List[str] = Field(None,description="List of documents urls to ingest")
    text:str= Field(None,description="Text to ingest")
    description:str=Field("", description="Description of the data folder content")
        
class UpdateDocsOfDataFolderForApi(BaseModel):
    nsId: str = Field("", description="Id of the Data Folder to Update")
    docs:List[str] = Field(None,description="List of documents names or URLs to ingest")
    text:str= Field(None,description="Text to ingest")
    action: str=Field("add", description="add to Add documents to data folder, replace to delete old content of data folder.")