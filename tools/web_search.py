from typing import Dict, Any
from langchain.tools import tool
from services.web_serach_service import run_web_search


@tool
def web_search(query: str) -> Dict[str, Any]:
    """ Search the web for information. """
    return run_web_search(query)
