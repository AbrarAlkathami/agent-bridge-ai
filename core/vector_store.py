# app/core/vector_store.py

from pathlib import Path
import duckdb

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DuckDB
from core.config import settings

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = STORAGE_DIR / "rag.duckdb"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=settings.openai_api_key)

_connection: duckdb.DuckDBPyConnection | None = None
_vector_store: DuckDB | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        _connection = duckdb.connect(str(DB_PATH))
    return _connection


def get_vector_store() -> DuckDB:
    global _vector_store
    if _vector_store is None:
        _vector_store = DuckDB(
            connection=get_connection(),
            embedding=embeddings,
            table_name="document_chunks",
        )
    return _vector_store


def close_connection() -> None:
    global _connection, _vector_store
    if _connection is not None:
        _connection.close()
        _connection = None
        _vector_store = None