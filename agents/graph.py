from typing import TypedDict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agents.model import model_provider


class RouteDecision(BaseModel):
    route: Literal["rag", "tools"]


class GraphState(TypedDict):
    query: str
    route: str


_ROUTER_PROMPT = (
    "You are a routing agent. Classify the user query into exactly one category:\n\n"
    '- "rag": The user is asking about uploaded documents, files, or stored knowledge base content.\n'
    '  Examples: "What does the document say about X?", "Summarize the file", "Find info in my docs"\n\n'
    '- "tools": The user needs current/real-time information, web search, or a structured research plan.\n'
    '  Examples: "What happened today?", "Search for Y", "Create a research plan about Z"\n\n'
    "Respond with ONLY the route: rag or tools"
)


async def supervisor_node(state: GraphState) -> dict:
    structured_model = model_provider.get_model().with_structured_output(RouteDecision)
    result = await structured_model.ainvoke([
        SystemMessage(content=_ROUTER_PROMPT),
        HumanMessage(content=state["query"]),
    ])
    return {"route": result.route}


def _build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", END)
    return builder.compile()


supervisor_graph = _build_graph()
