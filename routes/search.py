from fastapi import APIRouter
from langchain_core.messages import AIMessage, ToolMessage

from agents.rag_agent import rag_agent
from routes.schemas import QueryRequest

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query")
async def query_documents(request: QueryRequest) -> dict:
    answer = ""
    chunks = []

    for step in rag_agent(request.query):
        msg = step["messages"][-1]

        if isinstance(msg, AIMessage):
            answer = msg.content

        elif isinstance(msg, ToolMessage):
            docs = msg.artifact if hasattr(msg, "artifact") and msg.artifact else []
            chunks = [
                {"content": doc.page_content, "source": doc.metadata.get("source", "unknown"), "metadata": doc.metadata}
                for doc in docs
            ]

    return {"answer": answer, "chunks": chunks}
