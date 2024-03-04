from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class UserDb(BaseModel):
    userfrontendid:str
    password:str # hashed
    creationDate: Optional[datetime]
    email:str
    role:str # (superadmin, admin, client)
    name: Optional[str] 

    def __init__(self, userfrontendid:str, password:str, email:str, role:str, name: str,creationdate:datetime=datetime.now()):
        super().__init__(
            userfrontendid=userfrontendid,
            password=password,
            creationDate=creationdate or datetime.now(),
            email=email,
            role=role, #(superadmin, admin, user)
            name=name or ""
        )


            
class SessionDb(BaseModel):
    namespaceid:str = Field(..., description="The ID of the Namespace subject of this conversation Session")
    chathistory: List[str] = Field(..., description="The chat history for the session")
    chathistorylastdate: datetime = Field(default=datetime.now(), description="The date of the last saved chat history")
    startdate: datetime = Field(default=datetime.now(), description="The date the session was started")
    userfrontendid: str = Field(..., description="The Frontend ID of the user associated with the session")

    def __init__(self, namespaceid:str, chathistory: List[str], chathistorylastdate: datetime,
                 startdate: datetime,userfrontendid: str):
        super().__init__(
            namespaceid=namespaceid,
            chathistory=chathistory,
            chathistorylastdate=chathistorylastdate,
            startdate=startdate,
            userfrontendid=userfrontendid,
        )
        
        
class NamespaceDb(BaseModel):
    
    creationDate: datetime = Field(default=datetime.now(), description="The date the namespace was created")
    indexname: str = Field(..., description="The name of the index associated with the namespace in Pinecone")
    nsname: str = Field(..., description="The name of the namespace")
    nsdescription:str=Field(default="", description="The description of the namespace")
    pineconeName:str=Field(..., description="The Pinecone name of the namespace")
    userfrontendid:str=Field(..., description="User having created the Namespace")
    def __init__(self, indexname: str, nsname: str,  nsdescription: str,pineconeName:str, userfrontendid:str, creationDate: datetime = None):
        super().__init__(
            nsdescription=nsdescription,
            indexname=indexname,
            nsname=nsname,
            pineconeName=pineconeName,
            userfrontendid=userfrontendid,
            creationDate=creationDate or datetime.now(),
        )

