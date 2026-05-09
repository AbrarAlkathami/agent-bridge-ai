from langchain_core.messages import HumanMessage, SystemMessage
from agents.model import model_provider

_SYSTEM_PROMPT = """You are an AI research planner.

Create a practical research plan based on the user's topic and goal.
Do not return a generic fixed plan.
The plan must be specific to the topic.

Return:
- topic
- goal
- key research questions
- useful search queries
- step-by-step research actions
- final output format"""


async def dynamic_research_plan(topic: str, goal: str) -> str:

    model = model_provider.get_model()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Topic: {topic}\nGoal: {goal}"),
    ]
    response = await model.ainvoke(messages)
    return response.content
