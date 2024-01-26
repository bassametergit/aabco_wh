import json
from fastapi import APIRouter, Depends,  HTTPException
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
from apis_models import  SessionForUserSessions, DataFoldersForGetNamespaces,  UserForApi, DataFolderForApi, UpdateDocsOfDataFolderForApi
from chains_functions import answer_one_session_question
from ingest import ingest_urls_and_text_to_pinecone, normalize_string, verify_filenames_before_ingestion, add_string_to_pinecone
from db_models import NamespaceDb, SessionDb, UserDb
from data_access import Data_Access
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from typing import List
from dependencies import authorize_user
from internal.security_service import create_access_token, get_openai_and_pinecone_keys
import re
import threading


router = APIRouter(prefix="/jenny", tags=["Chatbot Jenny"])
    
@router.post("/create_user", status_code=201, description="Create a new User")
def create_user(user: UserForApi, _user=Depends(authorize_user)):
    """Creates a new User in Backend Db in table 'User'. If this User already exists (same userFrontendId), it only returns a new JWT for the User
       This Api can be called only with a superadmin JWT
    Args:
        user (UserForApi): Object with: userName (by default 'User'), userFrontendId (Mandatory string), email (Mandatory), role (one of superadmin, admin, user)
        _user (optional): user description after Bearer Jwt Auth. Defaults to Depends(authorize_user).

    Raises:
        HTTPException status_code=403, detail="Forbidden: Only Superadmin can create Users", when JWT isn't a superadmin one
        HTTPException status_code=400, detail="Bad Request: role should be one of (superadmin, admin, user)"
        HTTPException status_code=400, detail="Bad Request: Invalid email"
        HTTPException status_code=400, detail="Bad Request: User Exists but userFrontendId and email don't match"
        
    Returns:
    {
        "user_id": str,
        "user_jwt":str
    }
    """
    if (_user['role']!="superadmin"):
         raise HTTPException(status_code=403, detail="Forbidden: Only superadmin can create users")
   
    u=Data_Access.GetUserByFrontendId(userfrontendid=user.userFrontendId)
    if (u==None):
        if not user.role.lower() in {"superadmin", "admin", "user"}:
            raise HTTPException(status_code=400, detail="Bad Request: role should be one of (superadmin, admin, user)")
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        # Match the pattern against the email string
        match = re.match(pattern, user.email.lower()) # Returns True if the email is valid, False otherwise
        if match==None:
            m="Bad Request: Invalid email \'"+user.email+"\'"
            raise HTTPException(status_code=400, detail=m)
        newId=Data_Access.CreateUserDb(UserDb(userfrontendid=user.userFrontendId,  name=user.userName, email=user.email, role=user.role.lower()))
    else:
        if (u['email']!=user.email.lower()):
            raise HTTPException(status_code=400, detail="Bad Request: User Exists but userFrontendId and email don't match")
        newId=str(u['_id'])
    
    access_token = create_access_token(data={"username": user.userName, "email":user.email, "userfrontendid":user.userFrontendId,
                                             "backendid":newId, "role":user.role.lower()})
    return JSONResponse(content={
            "user_id": newId,
            "user_jwt":access_token
    })
    
@router.put("/update_user_role", status_code=201, description="Update User Role to one of (superadmin, admin, user)")
def update_user_role(userFrontendId:str, newRole:str, _user=Depends(authorize_user)):
    """
    Update User Role to one of (superadmin, admin, user) and generates new JWT
    This Api can be called only with a superadmin JWT
    
    Args:
        userFrontendId (str): The Frontend user Id
        newRole (str): one of (superadmin, admin, user)
        _user (optional): user description after Bearer Jwt Auth. Defaults to Depends(authorize_user).

    Raises:
        HTTPException: status_code=403, detail="Forbidden. Only superadmin can update User Role
        HTTPException: status_code=400, detail="Bad Request: role should be one of (superadmin, admin, user)

    Returns:
    {
        "updated": int, #(number of updated documents in Db (1 or 0))
        "user_jwt":str
    }
    """
    
    if (_user['role']!="superadmin"):
         raise HTTPException(status_code=403, detail="Forbidden. Only superadmin can update User Role")
    if not newRole.lower() in {"superadmin", "admin", "user"}:
            raise HTTPException(status_code=400, detail="Bad Request: role should be one of (superadmin, admin, user)")
    updated=Data_Access.UpdateUserRole(userFrontendId=userFrontendId,newRole=newRole.lower())
    if (updated==1):
        u=Data_Access.GetUserByFrontendId(userfrontendid=userFrontendId)
        access_token = create_access_token(data={"username": u['name'], "email":u['email'], "userfrontendid":userFrontendId,
                                             "backendid":str(u['_id']), "role":u['role'].lower()})
    else:
        access_token =""
    return JSONResponse(content={
        "updated": updated,"user_jwt":access_token
    })

@router.get("/get_user_details", status_code=201, description="Get User Details")
def get_user_details(userFrontendId: str, _user=Depends(authorize_user)):
    """
API Endpoint: /get_user

Description:
    Retrieve detailed information about a user based on their frontend ID. Only authorized superadmins and admins are allowed to access this endpoint.

Parameters:
    userFrontendId (str): The frontend ID of the user for whom details are to be retrieved.

Dependencies:
    _user (dict): Dependency for authorization, containing information about the user's role.

Returns:
    dict: A dictionary containing the following user details:
        - creationDate (str): The creation date of the user.
        - name (str): The name of the user.
        - email (str): The email address of the user.
        - role (str): The role of the user.

Raises:
    HTTPException:
        - 403 Forbidden: If the caller user's role is neither 'superadmin' nor 'admin'.
        - 404 Not Found: If the user with the specified frontend ID does not exist.

Note:
    This endpoint requires the user to have 'superadmin' or 'admin' role for access.

Example:
    Request:
        GET /get_user?userFrontendId=12345

    Response:
        {
            "creationDate": "2023-08-20",
            "name": "John Doe",
            "email": "john@example.com",
            "role": "user",
        }
"""

    if (_user['role']!="superadmin") and (_user['role']!="admin"):
         raise HTTPException(status_code=403, detail="Forbidden. Only superadmin and admin can Get User details")
    u=Data_Access.GetUserByFrontendId(userfrontendid=userFrontendId)
    if (u==None):
        raise HTTPException(status_code=404, detail="Inexistent User")
    return { 
            "creationDate":u['creationDate'],
            "name":u['name'], "email":u['email'],"role":u['role']
    }

    
@router.post("/new_namespace", status_code=201, description="Create a new Data Folder from Data File(s). If the Data Folder already exists, it throws Error 409")
def create_namespace(ns: DataFolderForApi, _user=Depends(authorize_user)):
    """
API Endpoint: /new_namespace

Description:
    Create a new Data Folder from Data File(s). If the Data Folder already exists, it throws Error 409. This operation is restricted to authorized users with roles other than 'user'.
    It also creates a new namespace on Pinecone.
Parameters:
    ns (DataFolderForApi): An object containing information for creating the new Data Folder.
        - nsName (str): A name for the Data Folder.
        - indexName (str) : Index name in which this namespace should be created  in Pinecone.
        - docs (List[str]): List of document names or URLs to ingest (optional).
        - text (str): text to ingest (optional).
        - description (str): Description of the data folder content.


    _user (dict): Dependency for authorization, containing information about the user's role.

Returns:
    JSONResponse: A JSON response containing the MongoDB ID of the newly created Data Folder.

Raises:
    HTTPException:
        - 400 Bad Request: If any input validation fails (e.g.empty 'docs' field).
        - 403 Forbidden: If the caller user's role is 'user', as users are not allowed to create Data Folders.
        - 409 Conflict: If a Data Folder with the same name already exists.

Note:
    - This endpoint requires the caller user to have a role other than 'user' for access.
    - As ingestion on Pinecone could take relatively long time.


Response:
        {
            "datafolder_id": "new_data_folder_id"
        }
"""

    if (_user['role']=="user") :
         raise HTTPException(status_code=403, detail="Forbidden. user cannot create Data Folder")
    if (ns.docs==None or ns.docs==[]) and (ns.text==None or ns.text==""):
        raise HTTPException(status_code=400, detail="Provide docs list and/or text to ingest.")
    ns.nsName=normalize_string(ns.nsName) #remove non-ascii and others characters
    nsdb=Data_Access.GetNamespaceByName(ns.nsName)
    if nsdb!=None:
        raise HTTPException(status_code=409, detail="Data Folder already exists. Run /update_datafolder to Update a Data Folder or choose a new name.")
    if (ns.docs!=None) and (len(ns.docs)!=0):
        verify_filenames_before_ingestion(docs=ns.docs) #will generate exception if any filename or url isn't valid or is unsupported format to ingest
    newId=Data_Access.CreateNamespaceDb(NamespaceDb(indexname=ns.indexName,
                                        nsname=ns.nsName, 
                                        nsdescription=ns.description, pineconeName=ns.nsName+"_"+_user['userfrontendid'], creationDate=datetime.now())) 
    openai_key, pinecone_key, pinecone_env=get_openai_and_pinecone_keys()
    ingest_urls_and_text_to_pinecone(urls=ns.docs, chunkSize=300, chunkOverlap=30, ind_name=ns.indexName,
                                            nsname=ns.nsName+"_"+_user['userfrontendid'], delete_ns_if_exists=True, openaikey=openai_key,
                                            pineconekey=pinecone_key,pineconeenv=pinecone_env,text=ns.text)    
    return JSONResponse(content={
        "datafolder_id": newId
    })
    

@router.put("/update_namespace", status_code=201, description="Replace (action=replace) or Add (action =add) document(s) to an existing Data Folder. Throws Error 404 if the Data Folder doesn't exist.")
def update_namespace(ns: UpdateDocsOfDataFolderForApi, _user=Depends(authorize_user)):
    """
API Endpoint: /update_namespace

Description:
    Replace (action=replace) or Add (action=add) document(s) to an existing Data Folder. Throws Error 404 if the Data Folder doesn't exist.

Parameters:
    ns (UpdateDocsOfDataFolderForApi): An object containing information for updating documents in an existing Data Folder.
        - nsId (str): ID of the existing Data Folder.
        - docs (List[str]): List of updated document URLs to ingest (optional).
        = text" text to ingest (optional).
        - action (str): Action to perform: 'add' or 'replace'.

    _user (dict): Dependency for authorization, containing information about the user's role.

Returns:
    JSONResponse: A JSON response containing the ID of the updated Data Folder.

Raises:
    HTTPException:
        - 400 Bad Request: If any input validation fails (e.g.empty 'docs' field, action is neither add nor replace).
        - 403 Forbidden: If the user's role is 'user', as users are not allowed to update Data Folders.
        - 404 Not Found: If the provided nsId does not correspond to an existing Data Folder.

Note:
    - This endpoint requires the client to have a role other than 'user' for access.
    - As ingestion on Pinecone could take relatively long time.

Response:
        {
            "nsid": "updated_ns_id"
        }
"""
    if (_user['role']=="user"):
        raise HTTPException(status_code=403, detail="Forbidden. User cannot update Namespace")
    ns.action=ns.action.lower()
    if ns.action!="add" and ns.action!="replave":
        raise HTTPException(status_code=400, detail="action should be add or replace.")
    if not Data_Access.IsValidObjectId(ns.nsId):
        raise HTTPException(status_code=404, detail="Invalid Data Folder Id. Input Data Folder Id provided by Api /new_datafolder")
    nsdb=Data_Access.GetNamespace(ns.nsId)
    if nsdb==None:
        raise HTTPException(status_code=404, detail="Inexistent Namespace. Run /new_namespace to create a new Namespace.")
    if (ns.docs!=None) and (len(ns.docs)!=0):
        verify_filenames_before_ingestion(docs=ns.docs) #will generate exception if any filename or url isn't valid or is unsupported format to ingest
    delete=(ns.action=="replace")
    openai_key, pinecone_key, pinecone_env=get_openai_and_pinecone_keys()
    ingest_urls_and_text_to_pinecone(urls=ns.docs, chunkSize=300, chunkOverlap=30, ind_name=nsdb['indexname'],
                                            nsname=nsdb['nsname']+"_"+_user['userfrontendid'], delete_ns_if_exists= delete, openaikey=openai_key,
                                            pineconekey=pinecone_key,pineconeenv=pinecone_env, text=ns.text)
    return JSONResponse(content={
        "nsid": ns.nsId
    })

@router.get("/index_namespaces", status_code=200, description="Get the list of all existing namespaces in a given index")
def index_namespaces(indexName:str, _user=Depends(authorize_user)) -> List[DataFoldersForGetNamespaces]:
    """
    Get the list of data folders in an index.

    Parameters:
        indexName (str): Index Name in Pinecone.

    Returns:
        List[DataFoldersForGetNamespaces]: A list of data folders .

    Model Classes:
        - DataFoldersForGetNamespaces:
            id:str
            name: str 
            creationDate: datetime 
            description:str
    """
   
    if (_user['role']=="user") :
        raise HTTPException(status_code=403, detail="Forbidden. User cannot Get List of Data Folders")
    namespaces= Data_Access.get_index_namespaces(indname=indexName)
    nsss:List[DataFoldersForGetNamespaces]=[]
    for ns in namespaces:
        nsss.append(DataFoldersForGetNamespaces(id=str(ns['_id']),name=ns['nsname'],creationDate=ns['creationDate'],description=ns['nsdescription']))
    return nsss

@router.get("/user_sessions", status_code=200, description="Get the list of all user non-empty sessions for a given namespace")
def user_sessions(userFrontendId:str, namespaceId:str, _user=Depends(authorize_user)) -> List[SessionForUserSessions]:
    """
    Get the list of all user non-empty sessions for a given namespace.

    Parameters:
        userFrontendId (str): The Frontend User ID.
        namespaceId (str)L The namespace Id

    Returns:
        List[SessionForUserSessions]: A list of sessions.

    Model Classes:
        - SessionForUserSessions:
            sessionId: str
            title: str
            startDate: datetime
            chatHistory: List[List[str]]
            lastChatHistoryUpdate: datetime
    """
   
    if (_user['role']=="user") :
        raise HTTPException(status_code=403, detail="Forbidden. User cannot Get List of User sessions")
    sessions= Data_Access.get_user_sessions_for_namespaceid(userfrontendid=userFrontendId, namespaceid=namespaceId)
    sss:List[SessionForUserSessions]=[]
    for s in sessions:
        if len(s['chathistory'])>0:
            sss.append(SessionForUserSessions(sessionId=str(s['_id']),
                                              title=s['chathistory'][0][0], 
                                              startDate=s['startdate'],
                                              chatHistory=s['chathistory'],
                                          lastChatHistoryUpdate=s['chathistorylastdate']))
    return sss

@router.post("/new_session", status_code=201, description="Creation of a Session for a User")
def new_session(userFrontendId:str, namespaceId:str, _user=Depends(authorize_user))->str:
    """
    Create a  conversation session for a user and a given namespace.

   
    Parameters:
        userFrontendId (str): The Frontend User ID.
        namespaceId (str)L The namespace Id

    Returns:
        JSONResponse: A JSON response containing the session ID: {"sessionId": str}

    Raises:
        HTTPException:
            - 404 Not Found: If the provided namespaceId is invalid, or Inexistent user
    """
    if (_user['role']=="user") :
        raise HTTPException(status_code=403, detail="Forbidden. User cannot create session")
    if not Data_Access.IsValidObjectId(namespaceId):
        raise HTTPException(status_code=404, detail="Invalid Namespace Id. Input Namespace Id provided by Api /new_namespace")
    u=Data_Access.GetUserByFrontendId(userFrontendId)
    if u==None:
        raise HTTPException(status_code=404, detail="Invalid userFrontendId.")
    ns=Data_Access.GetNamespace(id=namespaceId)
    if ns==None:
        m="Namespace of Id "+ namespaceId+" Not Found."
        raise HTTPException(status_code=404, detail=m)
    newId=Data_Access.CreateSessionDb(SessionDb(namespaceid=namespaceId,chathistory=[],chathistorylastdate=datetime.now(),startdate=datetime.now(),userfrontendid=userFrontendId))
    return JSONResponse(content={"sessionId": newId})    

@router.post("/question", status_code=201, description="Ask a question for Chatbot Jenny")
def question(sessionId:str, q: str, _user=Depends(authorize_user)):
    """
    Ask a question to chatbot Jenny and retrieve an answer.
   
    Parameters:
        sessionId (str): The User Session.
        q (str): The question text.
    Returns:
        str: The answer.

    Raises:
        HTTPException:
            - 404 Not Found: If the provided session ID is invalid or inexistent.
    """
    if not Data_Access.IsValidObjectId(sessionId):
        raise HTTPException(status_code=404, detail="Invalid Session Id.")
    session=Data_Access.GetSession(sessionId)
    if session==None:
        raise HTTPException(status_code=404, detail="Inexistent Session.")
    ns=Data_Access.GetNamespace(session['namespaceid'])
    if ns==None:
        m="Inexistent Namespace of Id "+session['namespaceid']+". Maybe deleted after creation of this session creation." 
        raise HTTPException(status_code=404, detail=m)
    chath = [tuple(l) for l in session['chathistory']]
    openai_key, pinecone_key, pinecone_env=get_openai_and_pinecone_keys()
    answer, updated_chat_history=answer_one_session_question(query=q,pineconekey=pinecone_key,openaik=openai_key,
                                                             indexname=ns['indexname'],pineconeenv=pinecone_env,
                                                             pineconenamespace=ns['pineconeName'],
                                                             model="gpt-4-0314",questionAnsweringTemperature=0.9,maxTokens=3000,similarSourceDocuments=3,
                                                             chat_history=chath)
    Data_Access.UpdateChatHistory(sessionid=sessionId, history=updated_chat_history)
    
    # add question and answer to Pinecone namespace
    text_to_ingest = "user question ("+str(datetime.now())+"): " + q +"\n" + "your answer (GPT): " + answer
    thread = threading.Thread(target=add_string_to_pinecone, args=(text_to_ingest,300, 30, ns['indexname'],
                                            ns['pineconeName'], openai_key,
                                            pinecone_key,pinecone_env))
    thread.start()
    return {"answer":answer}

@router.get("/test", status_code=201, description="test connect")
def test():
    

    return { 
            "result":1,
    }