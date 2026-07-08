from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.llm_client import LLMClient
from app.retrieval import HybridRetriever
from app.schemas import Citation, JobProfile, MatchReport, ResumeProfile


class CareerState(TypedDict, total=False):
    job_description: str
    job_profile: JobProfile
    resume_profile: ResumeProfile
    citations: list[Citation]
    report: MatchReport
    trace_id: str
    errors: list[str]


class CareerAgentWorkflow:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = HybridRetriever()
        graph = StateGraph(CareerState)
        graph.add_node("extract_job", self.extract_job)
        graph.add_node("parse_resume", self.parse_resume)
        graph.add_node("retrieve_evidence", self.retrieve_evidence)
        graph.add_node("score_match", self.score_match)
        graph.add_node("review_report", self.review_report)
        graph.set_entry_point("extract_job")
        graph.add_edge("extract_job", "parse_resume")
        graph.add_edge("parse_resume", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "score_match")
        graph.add_edge("score_match", "review_report")
        graph.add_edge("review_report", END)
        self.graph = graph.compile()

    def run(self, job_description: str, top_k: int = 6) -> CareerState:
        trace_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        state: CareerState = {
            "job_description": job_description,
            "trace_id": trace_id,
            "errors": [],
        }
        result = self.graph.invoke(state)
        self._write_trace(trace_id, {
            "type": "agent_match",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "result": self._dump_state(result),
        })
        return result

    def extract_job(self, state: CareerState) -> CareerState:
        system = "你是招聘 JD 结构化抽取器。请抽取 required_skills、preferred_skills、responsibilities、keywords。"
        user = f"岗位 JD：\n{state['job_description']}"
        try:
            profile = LLMClient().chat_json(system, user, JobProfile)
        except Exception as exc:
            profile = self._fallback_job_profile(state["job_description"])
            state.setdefault("errors", []).append(f"extract_job fallback: {exc}")
        state["job_profile"] = profile
        return state

    def parse_resume(self, state: CareerState) -> CareerState:
        citations = self.retriever.retrieve("候选人的技能、项目经历、工具栈、AI 应用开发经验", top_k=6, source_type="resume")
        context = "\n".join(c.text_preview for c in citations)
        system = "你是简历结构化解析器。请抽取 skills、projects、tools、evidence。"
        user = f"简历资料：\n{context}"
        try:
            profile = LLMClient().chat_json(system, user, ResumeProfile)
        except Exception as exc:
            profile = self._fallback_resume_profile(context)
            state.setdefault("errors", []).append(f"parse_resume fallback: {exc}")
        state["resume_profile"] = profile
        return state

    def retrieve_evidence(self, state: CareerState) -> CareerState:
        job_profile = state.get("job_profile") or JobProfile()
        query = " ".join(job_profile.required_skills + job_profile.preferred_skills + job_profile.keywords)
        if not query.strip():
            query = state["job_description"]
        state["citations"] = self.retriever.retrieve(query, top_k=6)
        return state

    def score_match(self, state: CareerState) -> CareerState:
        job_profile = state.get("job_profile") or JobProfile()
        resume_profile = state.get("resume_profile") or ResumeProfile()
        candidate_terms = self._term_set(resume_profile.skills + resume_profile.tools + resume_profile.projects + resume_profile.evidence)
        required = job_profile.required_skills or job_profile.keywords
        required_terms = [term for term in required if term.strip()]
        matched = [term for term in required_terms if self._contains_any(term, candidate_terms)]
        missing = [term for term in required_terms if term not in matched]
        score = int(100 * len(matched) / max(len(required_terms), 1))
        state["report"] = MatchReport(
            score=score,
            matched_items=matched,
            missing_items=missing,
            evidence_sources=[f"{c.label}: {c.relative_path}" for c in state.get("citations", [])],
            recommendations=[
                "补充项目中的量化指标和技术难点。",
                "针对缺失技能增加一个可演示的小功能或实验报告。",
                "在 README 中加入架构图、接口文档和评测结果。",
            ],
            narrative="",
        )
        return state

    def review_report(self, state: CareerState) -> CareerState:
        report = state.get("report") or MatchReport()
        citations = state.get("citations", [])
        context = "\n".join(f"[{c.label}] {c.text_preview}" for c in citations)
        system = "你是 AI 应用开发实习面试辅导专家。请基于结构化匹配结果和证据，生成简洁、可信、可执行的中文报告。必须引用 [S1] 等证据。"
        user = f"匹配结果 JSON：\n{report.model_dump_json(force_ascii=False)}\n\n证据：\n{context}"
        try:
            narrative = LLMClient().chat_text(system, user, temperature=0.2)
        except Exception as exc:
            narrative = "LLM report generation failed. Please inspect structured MatchReport and citations."
            state.setdefault("errors", []).append(f"review_report fallback: {exc}")
        report.narrative = narrative
        state["report"] = report
        return state

    def _fallback_job_profile(self, text: str) -> JobProfile:
        terms = self._keyword_list(text)
        return JobProfile(required_skills=terms[:8], preferred_skills=terms[8:12], responsibilities=[], keywords=terms)

    def _fallback_resume_profile(self, text: str) -> ResumeProfile:
        terms = self._keyword_list(text)
        return ResumeProfile(skills=terms, projects=[], tools=terms, evidence=[text[:300]])

    def _keyword_list(self, text: str) -> list[str]:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{1,}|[\u4e00-\u9fff]{2,}", text)
        seen: list[str] = []
        for item in candidates:
            if item not in seen and len(item) <= 30:
                seen.append(item)
        return seen[:16]

    def _term_set(self, values: list[str]) -> set[str]:
        text = " ".join(values).lower()
        return set(re.findall(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", text))

    def _contains_any(self, term: str, candidate_terms: set[str]) -> bool:
        lowered = term.lower()
        return lowered in candidate_terms or any(t in lowered or lowered in t for t in candidate_terms)

    def _write_trace(self, trace_id: str, payload: dict[str, Any]) -> None:
        self.settings.traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.traces_dir / f"{trace_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _dump_state(self, state: CareerState) -> dict[str, Any]:
        dumped: dict[str, Any] = {}
        for key, value in state.items():
            if hasattr(value, "model_dump"):
                dumped[key] = value.model_dump()
            elif isinstance(value, list):
                dumped[key] = [item.model_dump() if hasattr(item, "model_dump") else item for item in value]
            else:
                dumped[key] = value
        return dumped