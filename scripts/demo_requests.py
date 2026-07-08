import requests

BASE_URL = "http://127.0.0.1:8010"


def show(title: str, response: requests.Response) -> None:
    print(f"\n=== {title} ===")
    print("status:", response.status_code)
    data = response.json()
    if isinstance(data, dict):
        for key in ["answer", "averages", "trace_id"]:
            if key in data:
                print(key, ":", data[key])
        if "results" in data:
            print("results:", data["results"][:2])
        if "report" in data:
            print("score:", data["report"].get("score"))
            print(data["report"].get("narrative"))
        if "citations" in data:
            print("citations:", len(data["citations"]))
    else:
        print(data)


def main() -> None:
    show("health", requests.get(f"{BASE_URL}/health"))
    show("documents", requests.get(f"{BASE_URL}/documents"))

    # Run this once after changing data. It calls embeddings and consumes API quota.
    # show("rebuild index", requests.post(f"{BASE_URL}/index/rebuild"))

    search_payload = {
        "query": "Python FastAPI RAG 向量数据库 AI 应用开发",
        "top_k": 5,
    }
    show("hybrid search", requests.post(f"{BASE_URL}/search/hybrid", json=search_payload))

    answer_payload = {
        "question": "请基于资料判断候选人是否适合 AI 应用开发实习，并给出证据。",
        "top_k": 5,
    }
    show("answer", requests.post(f"{BASE_URL}/answer", json=answer_payload))

    job_description = """
AI 应用开发实习生：熟悉 Python 和 FastAPI，了解 RAG、Embedding、向量数据库、Prompt Engineering，
能够调用 OpenAI-compatible API，有 Agent、知识库问答或 AI 产品项目经验优先。
"""
    show("agent match", requests.post(f"{BASE_URL}/agent/match-job", json={"job_description": job_description, "top_k": 6}))
    show("eval", requests.post(f"{BASE_URL}/eval/run"))


if __name__ == "__main__":
    main()