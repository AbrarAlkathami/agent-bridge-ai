from langchain_core.tools import tool

from services.research_planner_service import dynamic_research_plan as _build_plan


@tool(return_direct=True)
async def research_planner(topic: str, goal: str) -> str:
    """Create a structured research plan only when the user explicitly asks to plan or organize research.
    Do NOT use this for answering questions or searching for information — use web_search instead.
    """
    return await _build_plan(topic, goal)
