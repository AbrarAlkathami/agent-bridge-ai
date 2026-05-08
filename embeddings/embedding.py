from langchain.tools import tool
from langchain_core.documents import Document

from core.vector_store import get_vector_store


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple[str, list[Document]]:
    """Retrieve information from indexed documents to help answer a query."""
    retrieved_docs = get_vector_store().similarity_search(query, k=4)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
