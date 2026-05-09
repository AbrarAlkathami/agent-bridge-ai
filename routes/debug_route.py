from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from agents.rag_agent import rag_agent
from core.config import settings
from routes.schemas import (
    DebugChunkResult,
    DebugRagRequest,
    DebugRagResponse,
    DebugSearchRequest,
    DebugSearchResponse,
)
from services.retrieval_service import inspect_database, search_similar_chunks

router = APIRouter(tags=["debug"])


def _require_debug() -> None:
    if not settings.debug_mode:
        raise HTTPException(status_code=404, detail="Not found.")


@router.post("/debug/search", response_model=DebugSearchResponse)
async def debug_search(request: DebugSearchRequest) -> DebugSearchResponse:
    _require_debug()
    chunks = await run_in_threadpool(search_similar_chunks, request.query, request.k)
    return DebugSearchResponse(
        query=request.query,
        results=[DebugChunkResult(**c) for c in chunks],
    )


@router.get("/debug/db")
async def inspect_db() -> dict:
    _require_debug()
    return await run_in_threadpool(inspect_database)


@router.post("/debug/rag")
async def debug_rag(request: DebugRagRequest) -> DebugRagResponse:
    return StreamingResponse(
        rag_agent(request.query), 
        media_type="text/event-stream" # Use event-stream for better frontend handling
    )