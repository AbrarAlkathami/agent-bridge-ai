
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from agents.tools_agent import tools_agent
from routes.schemas import AgentResponse

router = APIRouter(prefix="/agent", tags=["agents"])


@router.get("/run", response_model=AgentResponse)
async def run_agent(question: str) -> AgentResponse:
    result = await run_in_threadpool(tools_agent, question)
    return AgentResponse(output=result["output"])
