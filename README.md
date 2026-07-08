# AI Career Agent Platform

AI Career Agent Platform 是一个进阶版 Python AI 应用项目。它不再只是普通 RAG 问答，而是一个面向 AI 应用开发实习求职场景的 Agentic RAG 系统。

项目能力包括：

- 使用 Qdrant 本地向量库保存文档 chunk 和 embedding。
- 使用 dense vector search + BM25 的 hybrid retrieval 检索资料。
- 使用 citations 返回回答依据，降低“凭空编造”的风险。
- 使用 LangGraph 编排多步骤 Agent workflow：JD 抽取、简历解析、证据检索、匹配评分、报告审查。
- 使用本地评测脚本衡量 keyword recall、source type hit、citation coverage、latency。
- 使用 FastAPI 封装为可演示的后端服务。

## 技术栈

Python, FastAPI, LangGraph, Qdrant, rank-bm25, OpenAI SDK, aihubmix, Pydantic

## 启动方式

1. 在 .env 填写你的 api key

2. 启动 API：

```powershell
cd "...\ai-career-agent-platform"
D:\an\envs\all-in-rag\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

3. 打开中文前端：`r`n`r`n```text`r`nhttp://127.0.0.1:8010/`r`n````r`n`r`n也可以打开 Swagger 接口文档：`r`n`r`n```text`r`nhttp://127.0.0.1:8010/docs`r`n```

4. 首次运行或修改 data 后，调用：

```text
POST /index/rebuild
```

5. 演示接口：

```text
POST /search/hybrid
POST /answer
POST /agent/match-job
POST /eval/run
```

## 学习文档

详细学习文档在：

```text
docs/LEARNING_GUIDE.md
```
