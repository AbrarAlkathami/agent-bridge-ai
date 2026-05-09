
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from routes.schemas import (
    WebSearchResponse,
    ResearchPlannerRequest,
    ResearchPlannerResponse,
)
from services.web_serach_service import run_web_search
from services.research_planner_service import dynamic_research_plan

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/web-search", response_model=WebSearchResponse)
async def web_search_route(query: str) -> WebSearchResponse:
    result = await run_in_threadpool(run_web_search, query)
    return WebSearchResponse(result=result)


@router.post("/research-planner", response_model=ResearchPlannerResponse)
async def research_planner_route(body: ResearchPlannerRequest) -> ResearchPlannerResponse:
    plan = await dynamic_research_plan(body.topic, body.goal)
    return ResearchPlannerResponse(plan=plan)


