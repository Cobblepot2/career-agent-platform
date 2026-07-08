# AI Career Agent Platform 学习文档

这份文档是给你自己学习和复盘用的。它不是项目 README 的重复，而是按“为什么这样设计、代码在哪里、应该怎么读、怎么做实验、面试怎么讲”的顺序来拆解整个项目。

## 0. 这个项目要解决什么问题

旧项目 `AI Internship Copilot` 解决的是：

```text
我能不能用 Python + FastAPI + LlamaIndex 做出一个基础 RAG 问答系统？
```

新项目 `AI Career Agent Platform` 解决的是：

```text
我能不能把 RAG 做得更像真实 AI 应用工程：可检索、可解释、可编排、可评测、可演示？
```

它的目标不是再做一个聊天机器人，而是做一个面向 AI 应用开发实习求职的 Agentic RAG 系统。

核心变化：

```text
基础 RAG
  文档 -> embedding -> 向量索引 -> 问答

进阶 Agentic RAG
  文档 -> metadata -> Qdrant 向量库 -> hybrid retrieval -> citations
  岗位 JD -> LangGraph 工作流 -> 结构化分析 -> 匹配报告 -> 评测闭环
```

## 1. 项目结构

```text
ai-career-agent-platform/
  app/
    config.py            读取环境变量和项目路径
    schemas.py           Pydantic 数据结构
    llm_client.py        aihubmix / OpenAI-compatible API 客户端
    document_store.py    文档读取、切分、embedding、Qdrant 建库
    retrieval.py         hybrid retrieval 和 citation answer
    workflow.py          LangGraph Agent 工作流
    evaluation.py        本地 RAG 评测
    main.py              FastAPI 接口层
  data/
    resume/              简历资料
    jobs/                岗位 JD
    projects/            项目说明、学习笔记
    notes/               其他补充资料
  indexes/
    chunks.json          chunk、metadata、embedding 的本地副本
  vector_store/
    qdrant/              Qdrant 本地向量库文件
  traces/
    *.json               每次 answer / agent 运行的 trace
  eval/
    golden_questions.jsonl  评测问题集
  scripts/
    demo_requests.py     一键演示请求脚本
    run_reindex.py       重建索引脚本
  docs/
    LEARNING_GUIDE.md    本学习文档
    RESUME_VERSION.md    简历写法
```

你读代码时不要从 `main.py` 一口气读到底。推荐顺序：

```text
config.py -> schemas.py -> document_store.py -> retrieval.py -> workflow.py -> evaluation.py -> main.py
```

## 2. 技术栈为什么这样选

### FastAPI

FastAPI 负责把 Python 能力变成 HTTP API。

如果没有 FastAPI，你的 AI 能力只能在 Notebook 里运行；有了 FastAPI，前端、Swagger、脚本、其他服务都可以调用。

本项目中的接口：

```text
GET  /health
GET  /documents
POST /index/rebuild
POST /search/hybrid
POST /answer
POST /agent/match-job
POST /eval/run
```

### Qdrant

Qdrant 是向量数据库。本项目使用本地模式，不需要 Docker 服务也能跑。

它解决的问题：

```text
embedding 不是普通文本，不能像字符串一样搜索。
Qdrant 可以保存向量，并根据相似度找出最相关的 chunk。
```

### BM25 / rank-bm25

只靠 embedding 有时会漏掉关键词，比如库名、岗位要求、英文缩写。

BM25 是传统关键词检索方法，适合精确匹配：

```text
FastAPI
Qdrant
RAG
OpenAI-compatible API
```

本项目把 dense vector search 和 BM25 合并，叫 hybrid retrieval。

### LangGraph

LangGraph 用来编排 Agent 工作流。

不用 LangGraph 时，很多逻辑会变成一大段 prompt：

```text
请分析岗位、分析简历、找证据、打分、生成建议……
```

用 LangGraph 后，流程被拆成多个节点：

```text
extract_job -> parse_resume -> retrieve_evidence -> score_match -> review_report
```

这样更像真实工程：每一步可以单独调试、保存中间结果、失败后定位问题。

### Pydantic

Pydantic 用来定义结构化数据。

比如：

```text
JobProfile
ResumeProfile
MatchReport
Citation
```

这能避免 AI 输出一堆散乱文本，让后端可以稳定处理字段。

## 3. 环境变量

`.env.example` 内容：

```env
AIHUBMIX_API_KEY=你的_aihubmix_key
AIHUBMIX_BASE_URL=https://aihubmix.com/v1
AIHUBMIX_LLM_MODEL=gpt-4o-mini
AIHUBMIX_EMBED_MODEL=text-embedding-3-small
AI_CAREER_AGENT_HOME=E:\实习相关\ai-career-agent-platform
```

真正运行前，复制成 `.env`，填入真实 key。

注意：不要把 `.env` 上传 GitHub。

## 4. 第一次运行顺序

1. 启动 API：

```powershell
cd "E:\实习相关\ai-career-agent-platform"
D:\an\envs\all-in-rag\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

2. 打开 Swagger：

```text
http://127.0.0.1:8010/docs
```

3. 检查服务：

```text
GET /health
```

4. 检查资料：

```text
GET /documents
```

5. 建索引：

```text
POST /index/rebuild
```

这一步会调用 embedding，会消耗 aihubmix 额度。

6. 测试检索：

```text
POST /search/hybrid
```

7. 测试回答：

```text
POST /answer
```

8. 测试 Agent：

```text
POST /agent/match-job
```

9. 测试评测：

```text
POST /eval/run
```

## 5. config.py 怎么读

`config.py` 的职责是集中管理路径和模型配置。

关键点：

```python
get_settings()
```

这个函数会：

```text
1. 找到项目目录
2. 读取 .env
3. 确定 data、indexes、vector_store、traces、eval 路径
4. 读取 aihubmix API key、base_url、模型名
```

为什么要单独做 `config.py`？

因为真实项目里不要在每个文件里到处写：

```python
Path("E:/...")
os.getenv("...")
```

集中管理配置，后面迁移目录或部署时才不会到处改代码。

## 6. schemas.py 怎么读

`schemas.py` 是项目的“数据合同”。

你重点看这些类：

```text
ChunkRecord
Citation
HybridSearchRequest
AnswerResponse
JobProfile
ResumeProfile
MatchReport
AgentMatchResponse
EvalResult
```

为什么它重要？

因为 AI 应用很容易变成：

```text
输入是一段字符串
输出也是一段字符串
中间全靠 prompt 猜
```

这样不稳定。

Pydantic 的作用是让数据结构明确：

```text
岗位画像有哪些字段？
简历画像有哪些字段？
匹配报告有哪些字段？
API 返回给前端什么格式？
```

面试时可以说：

```text
我用 Pydantic 对 LLM 结构化输出进行约束，减少后端处理自然语言结果的不确定性。
```

## 7. document_store.py：索引构建主线

这是 RAG 的数据准备层。

核心流程：

```text
list_documents
  查看 data/ 里有哪些文件

rebuild_index
  读取文件 -> 切 chunk -> 调 embedding -> 保存 chunks.json -> 写入 Qdrant
```

### 7.1 资料目录如何映射 metadata

文件放在：

```text
data/resume/
data/jobs/
data/projects/
data/notes/
```

代码会把第一层目录变成 `source_type`：

```text
resume   简历
jobs     岗位 JD
projects 项目材料
notes    其他笔记
```

这就是 metadata。

metadata 的价值：

```text
查询时可以只查简历，或只查岗位 JD。
例如 source_type="resume"。
```

### 7.2 chunk 是什么

大模型不能无限读取长文档，所以要把文档切成片段。

本项目默认：

```text
chunk_size = 900 字符
chunk_overlap = 120 字符
```

overlap 的作用是避免切分处丢上下文。

### 7.3 chunks.json 是什么

`indexes/chunks.json` 保存：

```text
chunk id
chunk text
metadata
embedding vector
```

虽然 Qdrant 也保存了向量，但本地 JSON 方便调试和评测。

### 7.4 Qdrant 保存了什么

Qdrant collection 叫：

```text
career_agent_chunks
```

每个 point 包含：

```text
vector: embedding
payload: chunk_id, text, source_type, file_name, relative_path, chunk_index
```

## 8. retrieval.py：高级检索怎么工作

这是本项目技术含量最核心的部分之一。

接口：

```python
HybridRetriever().retrieve(query, top_k=5, source_type=None)
```

流程：

```text
用户问题
  ↓
生成 query embedding
  ↓
Qdrant dense vector search
  ↓
BM25 keyword search
  ↓
分数归一化
  ↓
0.68 * dense + 0.32 * BM25
  ↓
返回 citations
```

### 8.1 为什么不只用向量检索

向量检索擅长语义相似。

例如：

```text
“后端接口开发” 和 “FastAPI REST API”
```

可能语义相近。

但如果你要精确查：

```text
Qdrant
RAG
OpenAI-compatible API
```

BM25 往往更稳。

### 8.2 为什么要 citations

没有 citations 的回答像这样：

```text
你很适合这个岗位。
```

有 citations 的回答像这样：

```text
你具备 Python、FastAPI 和 RAG 项目经验 [S1][S2]，但缺少 Qdrant 和评测经验 [S3]。
```

这会让项目更专业，因为答案可以追溯来源。

### 8.3 trace 是什么

每次 `/answer` 会在 `traces/` 里保存一个 JSON：

```text
question
latency_ms
citations
answer
```

这就是最简单的可观测性。

面试时可以说：

```text
我加入了 trace 记录，便于排查检索结果、模型输出和响应时间。
```

## 9. workflow.py：LangGraph Agent 工作流

这是本项目另一个核心点。

工作流：

```text
extract_job
  ↓
parse_resume
  ↓
retrieve_evidence
  ↓
score_match
  ↓
review_report
```

### 9.1 extract_job

输入岗位 JD，输出：

```text
required_skills
preferred_skills
responsibilities
keywords
```

也就是 `JobProfile`。

### 9.2 parse_resume

先检索简历资料：

```python
source_type="resume"
```

再让 LLM 抽取：

```text
skills
projects
tools
evidence
```

也就是 `ResumeProfile`。

### 9.3 retrieve_evidence

根据岗位要求，再去整个资料库检索证据。

这一步会把：

```text
简历、项目笔记、岗位资料
```

都纳入考虑。

### 9.4 score_match

这是程序逻辑，不完全交给 LLM。

它会比较：

```text
岗位要求 required_skills
候选人能力 skills/tools/projects/evidence
```

然后算出一个 0-100 的分数。

这一步的价值是：

```text
不是让 LLM 随口说“匹配度高”，而是有一个可解释的计算过程。
```

### 9.5 review_report

最后 LLM 负责把结构化结果变成自然语言报告。

这就是合理分工：

```text
程序负责结构和规则
LLM 负责理解和表达
```

## 10. evaluation.py：本地评测系统

这个模块是为了让项目从“能跑”升级到“能评估”。

评测文件：

```text
eval/golden_questions.jsonl
```

每一行是一个问题：

```json
{"question":"候选人是否适合 AI 应用开发实习？","expected_keywords":["Python","FastAPI","RAG"],"expected_source_types":["resume","projects"]}
```

评测指标：

```text
keyword_recall       检索结果是否包含预期关键词
source_type_hit      检索来源类型是否正确
citation_coverage    是否返回了 citations
latency_ms           检索耗时
```

这不是完整 Ragas，但思想类似：

```text
不要只看回答好不好看，还要看检索是否找到了正确上下文。
```

由于当前 Python 3.12 环境无法直接安装 ragas，本项目先用本地评测实现。如果以后新建 Python 3.11 环境，可以再接入 Ragas 的 faithfulness、context precision、context recall 等指标。

## 11. main.py：API 层

`main.py` 只做接口，不写太多业务逻辑。

接口解释：

### GET /health

检查项目是否启动、索引是否存在。

### GET /documents

列出 data 中的资料。

### POST /index/rebuild

重建索引：读取资料、切 chunk、调用 embedding、写入 Qdrant。

### POST /search/hybrid

只检索，不生成回答。

适合调试：

```text
我的问题到底检索到了哪些片段？
```

### POST /answer

检索 + LLM 回答。

### POST /agent/match-job

完整 Agent 工作流。

### POST /eval/run

运行本地评测。

## 12. 推荐学习顺序

### Day 1：跑通项目

目标：启动 FastAPI，打开 Swagger，理解每个接口。

任务：

```text
1. GET /health
2. GET /documents
3. 阅读 README.md
4. 阅读 docs/LEARNING_GUIDE.md 的 0-4 节
```

### Day 2：理解索引构建

目标：知道 `/index/rebuild` 做了什么。

阅读：

```text
app/document_store.py
```

笔记问题：

```text
chunk 是什么？
metadata 有什么用？
为什么要保存 chunks.json？
Qdrant collection 保存了什么？
```

### Day 3：理解 hybrid retrieval

阅读：

```text
app/retrieval.py
```

实验：

```text
用同一个 query 测试 top_k=3 和 top_k=8 的区别。
只查 source_type=resume。
只查 source_type=jobs。
```

笔记问题：

```text
dense search 和 BM25 各自解决什么问题？
为什么要做分数归一化？
为什么 answer 要返回 citations？
```

### Day 4：理解 Agent workflow

阅读：

```text
app/workflow.py
```

画出流程：

```text
extract_job -> parse_resume -> retrieve_evidence -> score_match -> review_report
```

笔记问题：

```text
哪些步骤适合 LLM？
哪些步骤适合程序规则？
为什么要保存 trace？
```

### Day 5：理解评测

阅读：

```text
app/evaluation.py
eval/golden_questions.jsonl
```

实验：

```text
新增 3 条 golden questions。
运行 /eval/run。
观察 keyword_recall 是否变化。
```

### Day 6：整理项目展示

任务：

```text
1. 把你的真实简历放入 data/resume/
2. 把真实 JD 放入 data/jobs/
3. 把项目说明放入 data/projects/
4. 重新 /index/rebuild
5. 用 /agent/match-job 生成一份报告
```

### Day 7：准备面试讲解

你要能讲清楚：

```text
1. 为什么基础 RAG 不够？
2. 为什么加入 Qdrant？
3. 为什么加入 hybrid retrieval？
4. LangGraph 每个节点负责什么？
5. 评测指标说明了什么？
6. 如果继续优化，你会做什么？
```

## 13. 面试中怎么讲这个项目

30 秒版本：

```text
我做了一个基于 Agentic RAG 的 AI 实习求职分析系统。系统把简历、岗位 JD 和项目笔记切分成 chunk 后生成 embedding，存入 Qdrant；检索时融合向量检索和 BM25 关键词检索，并返回 citations。业务上我用 LangGraph 把岗位要求抽取、简历解析、证据检索、匹配评分和报告生成拆成多个节点，最后通过 FastAPI 暴露接口，并加入本地评测脚本观察检索召回、来源命中和响应时间。
```

2 分钟版本：

```text
这个项目最开始是一个基础 RAG 求职助手，但我发现普通 RAG 项目竞争力不够，所以把它升级成 Agentic RAG 系统。数据层面，我按 resume、jobs、projects、notes 对资料做 metadata 标注，切分 chunk 后调用 embedding，并写入 Qdrant 本地向量库。检索层面，我没有只用向量检索，而是把 dense search 和 BM25 结合，解决语义检索和关键词精确匹配各自的不足。应用层面，我用 LangGraph 编排多步骤 Agent：先抽取 JD 要求，再解析简历能力，然后检索证据、计算匹配度、生成报告。工程层面，我用 FastAPI 暴露接口，并通过 traces 和 eval 脚本做调试和评测。
```

## 14. 下一步可以继续升级什么

优先级从高到低：

```text
1. 加入真正的 reranker 模型，例如 bge-reranker 或 LLM rerank。
2. 加入前端页面，展示匹配分数、证据来源和学习计划。
3. 加入 SQLite/PostgreSQL，保存每次岗位分析记录。
4. 引入 Python 3.11 环境并安装 Ragas，补 faithfulness/context precision/context recall。
5. 使用 Docker Compose 启动 FastAPI + Qdrant server。
6. 加入用户上传文件接口。
7. 加入流式输出。
```

## 15. 你应该提交到 GitHub 的内容

建议提交：

```text
app/
data/ 示例文件，不要放真实隐私简历
docs/
eval/
scripts/
README.md
.env.example
```

不要提交：

```text
.env
vector_store/
indexes/chunks.json 如果里面有真实简历内容
traces/ 如果里面有真实请求内容
__pycache__/
```

## 16. 这个项目相对旧项目的竞争力

旧项目能说明：

```text
我会做基础 RAG + FastAPI。
```

新项目能说明：

```text
我理解 AI 应用工程中的检索质量、结构化输出、Agent 编排、可解释性和评测闭环。
```

这就是它更适合写进简历的原因。