from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from starlette import status
from internal.security_service import decode_token

__oauth2_scheme__ = OAuth2PasswordBearer(tokenUrl="token")



def authorize_user(token: str = Depends(__oauth2_scheme__)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = decode_token(token)
        userfrontendid:str=payload.get("userfrontendid")
        if userfrontendid is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return payload
