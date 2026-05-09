from pydantic import BaseModel, Field


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    filename: str
    document_id: str
    content_type: str
    size_bytes: int


class UploadResponse(BaseModel):
    uploaded: list[DocumentResponse]
    message: str


class FileListItem(BaseModel):
    filename: str
    document_id: str
    chunk_count: int


class FileListResponse(BaseModel):
    files: list[FileListItem]


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# ── RAG ───────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RetrievedChunk(BaseModel):
    content: str
    source: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[RetrievedChunk] = []


# ── Tools ─────────────────────────────────────────────────────────────────────

class WebSearchResponse(BaseModel):
    result: dict


class ResearchPlannerRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)


class ResearchPlannerResponse(BaseModel):
    plan: str


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    output: str


# ── Debug ─────────────────────────────────────────────────────────────────────

class DebugSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class DebugChunkResult(BaseModel):
    content: str
    metadata: dict


class DebugSearchResponse(BaseModel):
    query: str
    results: list[DebugChunkResult]


class DebugRagRequest(BaseModel):
    query: str = Field(..., min_length=1)


class DebugRagResponse(BaseModel):
    query: str
    answer: str
