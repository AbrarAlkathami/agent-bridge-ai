from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent

from agents.model import model_provider
from tools.web_search import web_search
from tools.research_planner import research_planner


_TOOLS = [web_search, research_planner]

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools for web search and research planning. "
    "Use web_search for current or real-time information. "
    "Use research_planner to create structured research plans."
)


async def tools_agent(question: str):
    agent = create_agent(model=model_provider.get_model(), tools=_TOOLS, system_prompt=_SYSTEM_PROMPT)

    messages = {"messages": [HumanMessage(content=question)]}
    async for chunk in agent.astream(messages, stream_mode="messages"):
        msg, metadata = chunk

        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                yield ("token", msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        yield ("token", item["text"])
