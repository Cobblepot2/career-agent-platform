# 简历写法

## 项目名称

AI Career Agent Platform｜基于 Agentic RAG 的实习求职分析系统

## 技术栈

Python, FastAPI, LangGraph, Qdrant, Hybrid Retrieval, BM25, OpenAI-compatible API, Pydantic

## 推荐简历 Bullet

- 独立设计并实现基于 FastAPI、LangGraph 和 Qdrant 的 Agentic RAG 求职分析系统，将简历、岗位 JD、项目笔记构建为可检索知识库，支持岗位匹配、技能差距分析、证据检索和面试准备建议生成。
- 实现 dense vector search + BM25 的 hybrid retrieval 链路，并在回答中返回 citations，展示资料来源和相关片段，提升回答可解释性。
- 使用 LangGraph 将 JD 抽取、简历解析、证据检索、匹配评分和报告审查拆分为可追踪节点，保存 trace 便于调试和复盘。
- 设计本地 RAG 评测脚本，对 keyword recall、source type hit、citation coverage 和 latency 进行评估，形成从“能跑”到“可评估”的工程闭环。
- 使用 Pydantic 约束 JobProfile、ResumeProfile、MatchReport 等结构化输出，并通过 FastAPI 封装 /search/hybrid、/answer、/agent/match-job、/eval/run 等 REST API。

## 面试讲法

这个项目不是单纯调用大模型 API，而是围绕“AI 应用开发实习求职”设计了完整的 Agentic RAG 系统。我先把简历、岗位 JD、项目笔记切分成 chunk，生成 embedding 后存入 Qdrant，同时保留 BM25 关键词检索；查询时融合向量相关性和关键词相关性，再把证据交给 LangGraph 工作流。Agent 会先抽取 JD 要求，再解析简历能力，随后检索项目证据并生成可解释的匹配报告。最后我加入了本地评测脚本，观察检索召回、来源命中和响应时间。