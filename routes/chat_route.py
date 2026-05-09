import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.graph import supervisor_graph
from agents.rag_agent import rag_agent
from agents.tools_agent import tools_agent

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


@router.post("/")
async def chat(request: ChatRequest):
    async def event_stream():
        state = await supervisor_graph.ainvoke({"query": request.message})
        route = state["route"]

        yield f"data: {json.dumps({'type': 'route', 'content': route})}\n\n"

        if route == "rag":
            async for event_type, content in rag_agent(request.message):
                if event_type == "chunks":
                    yield f"data: {json.dumps({'type': 'chunks', 'content': content})}\n\n"
                elif event_type == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        else:
            async for event_type, content in tools_agent(request.message):
                if event_type == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
