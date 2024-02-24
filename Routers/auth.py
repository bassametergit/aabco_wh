from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm

from authentication.models import Token
from authentication.logic import Authentication

#router = APIRouter(prefix="", tags=["Auth"])


# @router.post(
#     "/token", 
#     response_model=Token, 
#     description="Login to get an access token JWT"
# )
# def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
#     return Authentication.sign_in(
#         form_data.username,
#         form_data.password
#     )
