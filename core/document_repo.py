from langchain_core.documents import Document

from core.vector_store import get_connection, get_vector_store


def chunk_count(document_id: str) -> int:
    return get_connection().execute(
        "SELECT COUNT(*) FROM document_chunks WHERE json_extract_string(metadata, '$.document_id') = ?",
        [document_id],
    ).fetchone()[0]


def delete_chunks(document_id: str) -> None:
    conn = get_connection()
    conn.execute(
        "DELETE FROM document_chunks WHERE json_extract_string(metadata, '$.document_id') = ?",
        [document_id],
    )
    conn.commit()


def insert_chunks(chunks: list[Document], ids: list[str]) -> None:
    get_vector_store().add_documents(chunks, ids=ids)
