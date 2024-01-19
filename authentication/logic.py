import socket
from typing import Optional
from fastapi import HTTPException, Request
from starlette import status

from authentication.models import Token
from internal.security_service import (create_access_token, verify_password, get_password_hash)


class Authentication:
    def __init__(self):
        pass

    @staticmethod
    def sign_in(
        username: str,
        password: str
        ) -> Token:
        invalid_creds_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        access_token = create_access_token(data={"username": username})
        return Token(access_token=access_token, token_type="bearer")
