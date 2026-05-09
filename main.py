import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from routes.chat_route import router as chat_router
from routes.tool_agent import router as tool_agent_router
from routes.upload_file_route import router as upload_router
from routes.tools import router as tools_router
from routes.debug_route import router as debug_router
from routes.search import router as search_router
from core.config import settings
from core.vector_store import close_connection

if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    close_connection()


app = FastAPI(lifespan=lifespan)
app.state.settings = settings
app.include_router(chat_router)
app.include_router(tool_agent_router)
app.include_router(upload_router)
app.include_router(tools_router)
app.include_router(debug_router)
app.include_router(search_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
