import tempfile
from pathlib import Path
from uuid import uuid7

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.concurrency import run_in_threadpool

from core.vector_store import get_vector_store
from routes.schemas import DocumentResponse, UploadResponse


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _load_documents(path: Path, filename: str, document_id: str) -> list[Document]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"document_id": document_id, "source": filename})
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def _index_documents(chunks: list[Document], document_id: str) -> int:
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    ids = [f"{document_id}:{i}" for i in range(len(chunks))]
    get_vector_store().add_documents(chunks, ids=ids)
    return len(chunks)


def _process_file(path: Path, filename: str, document_id: str) -> int:
    docs = _load_documents(path, filename, document_id)
    chunks = _split_documents(docs)
    if not chunks:
        raise ValueError(f"No chunks created from {filename}")
    return _index_documents(chunks, document_id)


async def process_uploaded_documents(files: list[UploadFile]) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded: list[DocumentResponse] = []

    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{file.content_type}' for '{file.filename}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
            )

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported extension: {ext}")

        contents = await file.read()
        document_id = str(uuid7())

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            await run_in_threadpool(_process_file, tmp_path, file.filename, document_id)
        finally:
            tmp_path.unlink(missing_ok=True)

        uploaded.append(DocumentResponse(
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(contents),
        ))

    return UploadResponse(
        uploaded=uploaded,
        message=f"{len(uploaded)} document(s) uploaded and stored successfully.",
    )
