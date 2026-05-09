import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from routes.schemas import DeleteResponse, FileListResponse, FileListItem, UploadResponse
from services.retrieval_service import get_document_file_path, list_documents
from services.upload_service import delete_document, process_uploaded_documents

router = APIRouter(prefix="/documents", tags=["documents"])


def _validate_document_id(document_id: str) -> str:
    try:
        uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document_id format.")
    return document_id


@router.post("/uploadfile/", response_model=UploadResponse)
async def upload_file(file: UploadFile) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=422, detail="Must be a file upload with a filename.")
    return await process_uploaded_documents([file])


@router.get("/", response_model=FileListResponse)
async def list_documents_route() -> FileListResponse:
    files: list[FileListItem] = await run_in_threadpool(list_documents)
    return FileListResponse(files=files)


@router.get("/{document_id}/preview")
async def preview_document(document_id: str) -> FileResponse:
    _validate_document_id(document_id)
    path = await run_in_threadpool(get_document_file_path, document_id)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), filename=path.name)


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document_route(document_id: str) -> DeleteResponse:
    _validate_document_id(document_id)
    found = await run_in_threadpool(delete_document, document_id)
    if not found:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document_id=document_id, message="Document deleted successfully.")
