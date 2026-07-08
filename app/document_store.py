from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.llm_client import LLMClient
from app.schemas import ChunkMetadata, ChunkRecord, DocumentRecord, SourceType


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


class DocumentStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunks_path = self.settings.indexes_dir / "chunks.json"
        self.settings.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[DocumentRecord]:
        records: list[DocumentRecord] = []
        if not self.settings.data_dir.exists():
            return records
        for path in sorted(self.settings.data_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                records.append(
                    DocumentRecord(
                        path=str(path.relative_to(self.settings.data_dir)),
                        source_type=self._source_type(path),
                        size_bytes=path.stat().st_size,
                    )
                )
        return records

    def rebuild_index(self, batch_size: int = 16) -> tuple[int, int]:
        documents = self._load_documents()
        chunks = self._chunk_documents(documents)
        if not chunks:
            raise ValueError("No chunks were created. Add .md, .txt, or .pdf files under data/.")

        llm = LLMClient()
        all_vectors: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            all_vectors.extend(llm.embed_texts([chunk.text for chunk in batch]))

        for chunk, vector in zip(chunks, all_vectors, strict=True):
            chunk.vector = vector

        self._save_chunks(chunks)
        self._build_qdrant(chunks)
        return len(documents), len(chunks)

    def load_chunks(self) -> list[ChunkRecord]:
        if not self.chunks_path.exists():
            raise FileNotFoundError("chunks.json not found. Run POST /index/rebuild first.")
        data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        return [ChunkRecord.model_validate(item) for item in data]

    def qdrant_client(self) -> QdrantClient:
        return QdrantClient(path=str(self.settings.vector_store_dir))

    def _build_qdrant(self, chunks: list[ChunkRecord]) -> None:
        client = self.qdrant_client()
        collection = self.settings.qdrant_collection
        vector_size = len(chunks[0].vector or [])
        existing = {c.name for c in client.get_collections().collections}
        if collection in existing:
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = []
        for idx, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=idx,
                    vector=chunk.vector,
                    payload={
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        **chunk.metadata.model_dump(),
                    },
                )
            )
        client.upsert(collection_name=collection, points=points)

    def _load_documents(self) -> list[tuple[Path, SourceType, str]]:
        docs: list[tuple[Path, SourceType, str]] = []
        for path in sorted(self.settings.data_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            text = self._read_file(path)
            if text.strip():
                docs.append((path, self._source_type(path), text))
        return docs

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install pypdf to read PDF files.") from exc
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return ""

    def _chunk_documents(self, docs: list[tuple[Path, SourceType, str]], chunk_size: int = 900, overlap: int = 120) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for path, source_type, text in docs:
            clean = re.sub(r"\s+", " ", text).strip()
            start = 0
            chunk_index = 0
            while start < len(clean):
                end = min(start + chunk_size, len(clean))
                piece = clean[start:end].strip()
                if piece:
                    chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{chunk_index}:{piece[:64]}").hex
                    chunks.append(
                        ChunkRecord(
                            id=chunk_id,
                            text=piece,
                            metadata=ChunkMetadata(
                                chunk_id=chunk_id,
                                source_type=source_type,
                                file_name=path.name,
                                relative_path=str(path.relative_to(self.settings.data_dir)),
                                chunk_index=chunk_index,
                            ),
                        )
                    )
                chunk_index += 1
                if end == len(clean):
                    break
                start = max(0, end - overlap)
        return chunks

    def _save_chunks(self, chunks: list[ChunkRecord]) -> None:
        payload = [chunk.model_dump() for chunk in chunks]
        self.chunks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _source_type(self, path: Path) -> SourceType:
        try:
            first = path.relative_to(self.settings.data_dir).parts[0].lower()
        except ValueError:
            return "unknown"
        if first in {"resume", "jobs", "projects", "notes"}:
            return first  # type: ignore[return-value]
        return "unknown"