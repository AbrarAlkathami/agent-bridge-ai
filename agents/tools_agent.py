from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from agents.model import get_model
from tools.web_search import web_search
from tools.research_planner import research_planner


_TOOLS = [web_search, research_planner]

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools for web search and financial calculations. "
    "Use web_search for current or real-time information. "
    "Use calculate_future_value to project investment growth. "
    "Use calculate_monthly_savings_goal to determine required monthly savings. "
    "annual_return_rate is always a decimal (e.g. 0.07 for 7%)."
)


@lru_cache(maxsize=1)
def _get_agent():
    return create_agent(
        model=get_model(),
        tools=_TOOLS,
        system_prompt=_SYSTEM_PROMPT,
    )


def tools_agent(question: str) -> dict:
    response = _get_agent().invoke({"messages": [HumanMessage(content=question)]})
    return {"output": response["messages"][-1].content}
