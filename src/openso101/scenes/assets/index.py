import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from .catalog import AssetCatalog
from ..models import file_digest

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def catalog_documents(catalog):
    documents = []
    for asset in catalog.list():
        metadata = json.loads((catalog.directory(asset.uid) / "metadata.json").read_text())
        tags = [item["name"] if isinstance(item, dict) else str(item) for item in metadata.get("tags", ())]
        text = " ".join((asset.name, str(metadata.get("description", "")), *tags))
        documents.append({"uid": asset.uid, "asset_sha256": asset.sha256, "text": text,
                          "metadata_sha256": file_digest(catalog.directory(asset.uid) / "metadata.json")})
    if not documents:
        raise ValueError("embedding 索引需要实际资产目录")
    return documents


def documents_digest(documents):
    return hashlib.sha256(json.dumps(documents, sort_keys=True).encode()).hexdigest()


class AssetIndex:
    def __init__(self, catalog: AssetCatalog, root: Path):
        self.catalog = catalog
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "index.json").read_text())
        if self.manifest["schema_version"] != 1:
            raise ValueError("不支持的 embedding 索引版本")
        documents = catalog_documents(catalog)
        if self.manifest["catalog_sha256"] != documents_digest(documents):
            raise ValueError("资产内容或 metadata 已改变，embedding 索引需要重新生成")
        path = self.root / "embeddings.npz"
        if file_digest(path) != self.manifest["embeddings_sha256"]:
            raise ValueError("embedding 文件 SHA256 不一致")
        with np.load(path, allow_pickle=False) as data:
            self.embeddings = data["embeddings"]
        if (self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(documents)
                or not np.isfinite(self.embeddings).all()
                or not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-5)):
            raise ValueError("embedding 数量、形状或数值无效")
        self.documents = documents
        self.model = SentenceTransformer(self.manifest["model"], revision=self.manifest["model_revision"],
                                         device="cpu", cache_folder=str(self.root.parent / "embedding_models"),
                                         trust_remote_code=False)

    @classmethod
    def build(cls, catalog, root: Path, *, model=DEFAULT_EMBEDDING_MODEL, revision=None):
        root = root.resolve()
        if root.exists():
            raise FileExistsError(root)
        documents = catalog_documents(catalog)
        revision = model_info(model, revision=revision).sha
        encoder = SentenceTransformer(model, revision=revision, device="cpu",
                                      cache_folder=str(root.parent / "embedding_models"), trust_remote_code=False)
        embeddings = encoder.encode([item["text"] for item in documents], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise RuntimeError("embedding 模型输出无效")
        root.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(root / "embeddings.npz", embeddings=embeddings)
        manifest = {"schema_version": 1, "model": model, "model_revision": revision,
                    "catalog_sha256": documents_digest(documents), "documents": documents,
                    "embeddings_shape": list(embeddings.shape), "embeddings_sha256": file_digest(root / "embeddings.npz"),
                    "index_source_sha256": file_digest(Path(__file__))}
        (root / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest

    def search(self, query: str, *, limit=8, minimum_similarity=.25):
        if not query.strip() or not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("检索需要非空 query 与正整数 limit")
        if not np.isfinite(minimum_similarity) or not -1 <= minimum_similarity <= 1:
            raise ValueError("minimum_similarity 需要位于 -1 到 1")
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        scores = self.embeddings @ vector
        return [self.documents[index] | {"similarity": float(scores[index])}
                for index in np.argsort(-scores)[:limit] if scores[index] >= minimum_similarity]
