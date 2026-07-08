from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.document_store import DocumentStore
from app.evaluation import LocalRAGEvaluator
from app.retrieval import HybridRetriever
from app.schemas import (
    AgentMatchResponse,
    AnswerRequest,
    AnswerResponse,
    EvalResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    JobMatchRequest,
    ReindexResponse,
)
from app.workflow import CareerAgentWorkflow

app = FastAPI(
    title="AI 求职 Agent 平台",
    description="面向 AI 应用开发实习准备的 Agentic RAG 系统。",
    version="0.1.0",
)

frontend_dir = get_settings().project_dir / "frontend"
app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


@app.get("/", include_in_schema=False)
def frontend_home() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    chunks_ready = (settings.indexes_dir / "chunks.json").exists()
    return {
        "status": "ok",
        "project_dir": str(settings.project_dir),
        "chunks_ready": chunks_ready,
        "collection": settings.qdrant_collection,
    }


@app.get("/documents")
def documents() -> dict:
    return {"documents": [doc.model_dump() for doc in DocumentStore().list_documents()]}


@app.post("/index/rebuild", response_model=ReindexResponse)
def rebuild_index() -> ReindexResponse:
    try:
        document_count, chunk_count = DocumentStore().rebuild_index()
        settings = get_settings()
        return ReindexResponse(
            status="ok",
            document_count=document_count,
            chunk_count=chunk_count,
            collection=settings.qdrant_collection,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search(request: HybridSearchRequest) -> HybridSearchResponse:
    try:
        results = HybridRetriever().retrieve(request.query, top_k=request.top_k, source_type=request.source_type)
        return HybridSearchResponse(query=request.query, results=results)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    try:
        answer_text, citations, trace_id = HybridRetriever().answer(
            request.question,
            top_k=request.top_k,
            source_type=request.source_type,
        )
        return AnswerResponse(answer=answer_text, citations=citations, trace_id=trace_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/match-job", response_model=AgentMatchResponse)
def match_job(request: JobMatchRequest) -> AgentMatchResponse:
    try:
        state = CareerAgentWorkflow().run(request.job_description, top_k=request.top_k)
        return AgentMatchResponse(
            job_profile=state["job_profile"],
            resume_profile=state["resume_profile"],
            report=state["report"],
            citations=state.get("citations", []),
            trace_id=state["trace_id"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/eval/run", response_model=EvalResponse)
def run_eval() -> EvalResponse:
    try:
        results, averages = LocalRAGEvaluator().run()
        return EvalResponse(results=results, averages=averages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc