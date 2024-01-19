
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
#from Routers.auth import router as auth_router
from Routers.chatbot import router as chatbot_router

from fastapi.openapi.utils import get_openapi


def create_app() -> FastAPI:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
    app = FastAPI(
        title="Aabco Chatbot Jenny- Work History API",
        description="An API for Aabco.",
        openapi_url="/swagger.json",
        middleware=middleware)
    #app.include_router(auth_router)
    app.include_router(chatbot_router)
    return app

app = create_app()

# Modify the OpenAPI schema to include JWT authentication
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Jenny",
        version="1.0.0",
        description="Aabco",
        routes=app.routes,
    )
    # Modify the Swagger UI configuration to include the JWT token
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

async def startup_event():
    print("Go")
    
app.add_event_handler("startup", startup_event)
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0",
        port=8000,
    )