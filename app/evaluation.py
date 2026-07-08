from __future__ import annotations

import json
import time

from app.config import get_settings
from app.retrieval import HybridRetriever
from app.schemas import EvalResult


class LocalRAGEvaluator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = HybridRetriever()

    def run(self) -> tuple[list[EvalResult], dict[str, float]]:
        questions = self._load_questions()
        results: list[EvalResult] = []
        for item in questions:
            start = time.perf_counter()
            citations = self.retriever.retrieve(item["question"], top_k=5)
            latency_ms = (time.perf_counter() - start) * 1000
            text = "\n".join(c.text_preview for c in citations).lower()
            expected_keywords = [str(k).lower() for k in item.get("expected_keywords", [])]
            expected_sources = set(item.get("expected_source_types", []))
            keyword_hits = sum(1 for keyword in expected_keywords if keyword in text)
            source_hits = sum(1 for c in citations if c.source_type in expected_sources)
            results.append(
                EvalResult(
                    question=item["question"],
                    latency_ms=round(latency_ms, 2),
                    retrieved_count=len(citations),
                    keyword_recall=round(keyword_hits / max(len(expected_keywords), 1), 4),
                    source_type_hit=round(source_hits / max(len(citations), 1), 4),
                    citation_coverage=1.0 if citations else 0.0,
                )
            )
        averages = self._averages(results)
        return results, averages

    def _load_questions(self) -> list[dict]:
        path = self.settings.eval_dir / "golden_questions.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _averages(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        keys = ["latency_ms", "keyword_recall", "source_type_hit", "citation_coverage"]
        averages = {}
        for key in keys:
            averages[key] = round(sum(getattr(result, key) for result in results) / len(results), 4)
        return averages