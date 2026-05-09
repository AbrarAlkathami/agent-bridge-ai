import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.tools_agent import tools_agent
from routes.schemas import QueryRequest

router = APIRouter(prefix="/agent", tags=["agents"])


async def event_stream(question: str):
    async for event_type, content in tools_agent(question):
        yield f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"


@router.post("/run")
async def run_agent(request: QueryRequest):
    return StreamingResponse(
        event_stream(request.query),
        media_type="text/event-stream"
    )
