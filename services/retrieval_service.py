from pathlib import Path

import duckdb

from core.vector_store import DB_PATH, get_connection, get_vector_store
from routes.schemas import FileListItem
from services.upload_service import FILES_DIR


def list_documents() -> list[FileListItem]:
    
    """ 
    Returns a list of documents with their filename, document_id, 
    and chunk count.
    """
    
    rows = get_connection().execute(
        """
        SELECT
            json_extract_string(metadata, '$.source')      AS filename,
            json_extract_string(metadata, '$.document_id') AS document_id,
            COUNT(*)                                        AS chunk_count
        FROM document_chunks
        GROUP BY 1, 2
        ORDER BY filename
        """
    ).fetchall()
    return [FileListItem(filename=r[0], document_id=r[1], chunk_count=r[2]) for r in rows]


def get_document_file_path(document_id: str) -> Path | None:
    
    """ 
    Given a document_id, returns the file path of the original uploaded document.
    Returns None if the document or file does not exist.
    """
    
    doc_dir = FILES_DIR / document_id
    
    if not doc_dir.exists():
        return None
    
    matches = [f for f in doc_dir.glob("*") if f.is_file()]
    
    return matches[0] if matches else None


def search_similar_chunks(query: str, k: int) -> list[dict]:
    
    """ 
    Performs a similarity search against the vector store and 
    returns the top k chunks.
    """
    
    docs = get_vector_store().similarity_search(query, k=k)
    
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


def inspect_database() -> dict:
    
    """ 
    Inspects the database and returns information about its structure and content.
    """
    
    if not DB_PATH.exists():
        return {
            "status": "not_initialized",
            "message": "Database does not exist yet. Upload documents first.",
        }

    con = duckdb.connect(str(DB_PATH))
    
    try:
        tables = [t[0] for t in con.sql("SHOW TABLES").fetchall()]
        result: dict = {"tables": tables}

        if "document_chunks" in tables:
            result["row_count"] = con.sql("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
            result["sample_rows"] = con.sql(
                "SELECT * FROM document_chunks LIMIT 3"
            ).fetchall()
    finally:
        con.close()

    return result
