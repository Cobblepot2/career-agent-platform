from __future__ import annotations

import json
import re
from typing import Any, Type

from openai import OpenAI
from pydantic import BaseModel

from app.config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.aihubmix_api_key or self.settings.aihubmix_api_key == "你的_aihubmix_key":
            raise ValueError("AIHUBMIX_API_KEY is missing. Copy .env.example to .env and fill your key.")
        self.client = OpenAI(
            api_key=self.settings.aihubmix_api_key,
            base_url=self.settings.aihubmix_base_url,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.settings.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def chat_text(self, system: str, user: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, model: Type[BaseModel]) -> BaseModel:
        raw = self.chat_text(
            system=system + "\n只输出 JSON，不要输出 Markdown 代码块。",
            user=user,
            temperature=0.1,
        )
        data = self._parse_json(raw)
        return model.model_validate(data)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))