
from datetime import datetime
from typing import List

from bson.objectid import ObjectId
from fastapi import HTTPException
from db_models import UserDb, NamespaceDb, SessionDb
from mongo_service import get_collection

db_name="aabco_jenny"

class Data_Access:
    
    def IsValidObjectId(id: str) -> bool:
        try:
            ObjectId(id)
            return True
        except:
            return False
    
    @staticmethod
    def CreateUserDb(u: UserDb): # returns user id
        collection =get_collection(db_name,'user')
        result = collection.insert_one(u.dict())
        return str(result.inserted_id)
    
    @staticmethod
    def UpdateUserRole(userFrontendId:str, newRole:str)->int: # returns 1 if updated, 0 if User not found
        collection = get_collection(db_name, 'user')
        result = collection.update_one({'userfrontendid': userFrontendId}, {'$set': {'role': newRole}})
        return result.modified_count
            
    @staticmethod
    def GetUserById(userId: str) -> UserDb:
        client_collection = get_collection(db_name, 'user')
        return client_collection.find_one({'_id': ObjectId(userId)})
     
    
    @staticmethod
    def GetUserByFrontendId(userfrontendid: str) -> UserDb:
        client_collection = get_collection(db_name, 'user')
        return client_collection.find_one({'userfrontendid': userfrontendid})
        
    @staticmethod
    def CreateNamespaceDb(ns: NamespaceDb): # returns namespace id
        collection = get_collection(db_name, 'namespace')
        result = collection.insert_one(ns.dict())
        return str(result.inserted_id)
     
    
        
    @staticmethod
    def CreateSessionDb(session: SessionDb): # returns Session id
        collection = get_collection(db_name, 'session')
        result = collection.insert_one(session.dict())
        return str(result.inserted_id)
    
    
    @staticmethod
    def GetSession(id):
        collection = get_collection(db_name, 'session')
        session = collection.find_one({"_id": ObjectId(id)})
        return session
    
    @staticmethod
    def GetNamespace(id:str):
        collection = get_collection(db_name, 'namespace')
        ns = collection.find_one({"_id": ObjectId(id)})
        return ns
     
    @staticmethod
    def GetNamespaceByName(name:str):
        collection = get_collection(db_name, 'namespace')
        ns = collection.find_one({"nsname": name.lower()})
        return ns
    
    @staticmethod
    def DeleteNamespace(id: str):
        user_collection = get_collection(db_name, 'namespace')
        result = user_collection.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 1:
            return True
        else:
            return False
    
    @staticmethod
    def DeleteNamespaceByName(name: str):
        collection = get_collection(db_name, 'namespace')
        result = collection.delete_one({"nsname": name.lower()})
        if result.deleted_count == 1:
            return True
        else:
            return False
    
    @staticmethod
    def UpdateChatHistory(sessionid:str, history: List[str]):
        collection = get_collection(db_name, 'session')
        collection.update_one(
        {'_id': ObjectId(sessionid)},
        {'$set': {'chathistory': history, 'chathistorylastdate': datetime.now()}})
    
    
    @staticmethod
    def get_namespaces() -> List[NamespaceDb]:
        ns_collection = get_collection(db_name, 'namespace')
        allns=ns_collection.find()
        namespaces:List[NamespaceDb]=[]
        for ns in allns:
            namespaces.append(ns)
        return namespaces
    
    @staticmethod
    def get_index_namespaces(indname):
        ns_collection = get_collection(db_name, 'namespace')
        query = {}
        query['indexname'] = indname
        allns=ns_collection.find(query)
        return allns
    

    # Function to retrieve sessions based on userfrontendid
    def get_user_sessions(userfrontendid: str) -> List[SessionDb]:
        collection = get_collection(db_name, 'session')
        query = {}
        query['userfrontendid'] = userfrontendid
        sessions = collection.find(query)
        return sessions
    
    # Function to retrieve sessions based on userfrontendid and namespaceid
    def get_user_sessions_for_namespaceid(userfrontendid: str, namespaceid:str):
        collection = get_collection(db_name, 'session')
        query = {"userfrontendid": userfrontendid, "namespaceid": namespaceid}
        sessions = collection.find(query)
        return sessions

    @staticmethod
    def get_user_session(user_id:str, session_id: str) -> SessionDb:
        user_collection = get_collection(db_name, 'user')
        session_collection = get_collection(db_name, 'session')
        user = user_collection.find_one({'_id': ObjectId(user_id)})
        one_session = session_collection.find_one({'_id': ObjectId(session_id)})
        if one_session!=None:
            return one_session
        else:
            raise HTTPException(status_code=404, detail="Session Not found.")
    
     
    @staticmethod
    def delete_all_empty_sessions():
        session_collection=get_collection(db_name, 'session')
        allsessions = session_collection.find()
        for s in allsessions:
            if s['chathistory']==None or s['chathistory']==[] or ((len(s['chathistory'])==1) and s['chathistory'][0][0].lower()=='hello' ) :
                session_collection.delete_one({'_id': s['_id']})
   