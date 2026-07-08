from __future__ import annotations

import math
import re
import time
import uuid
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.document_store import DocumentStore
from app.llm_client import LLMClient
from app.schemas import Citation, SourceType


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]", text.lower())
    return tokens


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = DocumentStore()

    def retrieve(self, query: str, top_k: int = 5, source_type: SourceType | None = None) -> list[Citation]:
        chunks = self.store.load_chunks()
        filtered = [c for c in chunks if source_type is None or c.metadata.source_type == source_type]
        if not filtered:
            return []

        dense_scores = self._dense_scores(query, top_k=max(top_k * 3, 10), source_type=source_type)
        bm25_scores = self._bm25_scores(query, filtered)
        dense_norm = normalize_scores(dense_scores)
        bm25_norm = normalize_scores(bm25_scores)

        combined: dict[str, float] = {}
        for chunk in filtered:
            cid = chunk.id
            combined[cid] = 0.68 * dense_norm.get(cid, 0.0) + 0.32 * bm25_norm.get(cid, 0.0)

        chunk_by_id = {chunk.id: chunk for chunk in filtered}
        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top_k]
        citations: list[Citation] = []
        for i, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = chunk_by_id[chunk_id]
            citations.append(
                Citation(
                    label=f"S{i}",
                    score=round(float(score), 4),
                    source_type=chunk.metadata.source_type,
                    file_name=chunk.metadata.file_name,
                    relative_path=chunk.metadata.relative_path,
                    text_preview=chunk.text[:500],
                )
            )
        return citations

    def answer(self, question: str, top_k: int = 5, source_type: SourceType | None = None) -> tuple[str, list[Citation], str]:
        trace_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        citations = self.retrieve(question, top_k=top_k, source_type=source_type)
        context = "\n\n".join(f"[{c.label}] {c.text_preview}" for c in citations)
        system = "你是一个严谨的 AI 求职分析助手。只能基于给定资料回答；如果资料不足，要明确说明。回答中请使用 [S1] 这样的引用标记。"
        user = f"问题：{question}\n\n可用资料：\n{context}"
        answer = LLMClient().chat_text(system=system, user=user, temperature=0.2)
        self._write_trace(trace_id, {
            "type": "answer",
            "question": question,
            "top_k": top_k,
            "source_type": source_type,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "citations": [c.model_dump() for c in citations],
            "answer": answer,
        })
        return answer, citations, trace_id

    def _dense_scores(self, query: str, top_k: int, source_type: SourceType | None) -> dict[str, float]:
        vector = LLMClient().embed_texts([query])[0]
        client = self.store.qdrant_client()
        query_filter = None
        if source_type is not None:
            query_filter = Filter(must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))])
        try:
            points = client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            points = []
        return {p.payload["chunk_id"]: float(p.score) for p in points if p.payload and "chunk_id" in p.payload}

    def _bm25_scores(self, query: str, chunks: list[Any]) -> dict[str, float]:
        corpus = [tokenize(chunk.text) for chunk in chunks]
        if not corpus:
            return {}
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(query))
        return {chunk.id: float(score) for chunk, score in zip(chunks, scores, strict=True)}

    def _write_trace(self, trace_id: str, payload: dict[str, Any]) -> None:
        self.settings.traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.traces_dir / f"{trace_id}.json"
        import json
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")