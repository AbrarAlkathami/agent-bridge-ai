from typing import Dict, Any
from langchain.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> Dict[str, Any]:
    """ Search the web for information. """
    client = TavilyClient()
    return client.search(query)
