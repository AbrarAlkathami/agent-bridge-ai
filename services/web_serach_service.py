from tavily import TavilyClient

from core.config import settings


def run_web_search(query: str) -> dict:
    client = TavilyClient(api_key=settings.tavily_api_key)
    return client.search(query)
