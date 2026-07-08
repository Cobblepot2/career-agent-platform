from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel


DEFAULT_HOME = r"E:\实习相关\ai-career-agent-platform"


class Settings(BaseModel):
    project_dir: Path
    data_dir: Path
    indexes_dir: Path
    vector_store_dir: Path
    traces_dir: Path
    eval_dir: Path
    qdrant_collection: str = "career_agent_chunks"
    aihubmix_api_key: str | None = None
    aihubmix_base_url: str = "https://aihubmix.com/v1"
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_dir = Path(os.getenv("AI_CAREER_AGENT_HOME", DEFAULT_HOME))
    env_path = project_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    project_dir = Path(os.getenv("AI_CAREER_AGENT_HOME", str(project_dir)))
    return Settings(
        project_dir=project_dir,
        data_dir=project_dir / "data",
        indexes_dir=project_dir / "indexes",
        vector_store_dir=project_dir / "vector_store" / "qdrant",
        traces_dir=project_dir / "traces",
        eval_dir=project_dir / "eval",
        aihubmix_api_key=os.getenv("AIHUBMIX_API_KEY"),
        aihubmix_base_url=os.getenv("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
        llm_model=os.getenv("AIHUBMIX_LLM_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("AIHUBMIX_EMBED_MODEL", "text-embedding-3-small"),
    )