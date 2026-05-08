# app/core/vector_store.py

from pathlib import Path
import duckdb

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DuckDB

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = STORAGE_DIR / "rag.duckdb"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

connection = duckdb.connect(str(DB_PATH))  # this creates rag.duckdb if it does not exist

vector_store = DuckDB(
    connection=connection,
    embedding=embeddings,
    table_name="document_chunks",
)


def get_vector_store() -> DuckDB:
    return vector_store