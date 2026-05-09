from langchain_core.tools import tool

from services.research_planner_service import fixed_research_plan as _build_plan


@tool
def research_planner(topic: str) -> str:
    """Create a structured research plan for a given topic.
    Use this tool when the user asks how to organize, plan, or structure research on any subject.
    """
    return _build_plan(topic)
