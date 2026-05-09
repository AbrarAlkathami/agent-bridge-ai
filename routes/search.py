import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agents.rag_agent import rag_agent
from routes.schemas import QueryRequest

router = APIRouter(prefix="/rag", tags=["rag"])


async def event_stream(query: str):
    async for event_type, content in rag_agent(query):
        yield f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"


@router.post("/query")
async def query_documents(request: QueryRequest):
    return StreamingResponse(
        event_stream(request.query),
        media_type="text/event-stream"
    )
