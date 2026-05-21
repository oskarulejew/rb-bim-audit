from __future__ import annotations
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from .io_utils import as_text

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

_TABLE_EXTS = {'.xlsx', '.xlsm', '.xls', '.csv'}
_EMBED_MODEL = 'text-embedding-3-small'
_MIN_SCORE = 0.28


def _file_hash(paths: list[Path]) -> str:
    h = hashlib.md5()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode())
        try:
            h.update(str(Path(p).stat().st_mtime_ns).encode())
        except Exception:
            pass
    return h.hexdigest()[:16]


def _chunks_from_file(path: Path) -> list[dict]:
    chunks = []
    try:
        if path.suffix.lower() == '.csv':
            sheets = {'': pd.read_csv(path, dtype=str).fillna('')}
        else:
            xl = pd.ExcelFile(path)
            sheets = {}
            for s in xl.sheet_names:
                try:
                    sheets[s] = xl.parse(s, dtype=str).fillna('')
                except Exception:
                    pass
    except Exception:
        return chunks

    for sheet, df in sheets.items():
        if df.empty or len(df.columns) < 2:
            continue
        df.columns = [as_text(c).strip() for c in df.columns]
        for _, row in df.iterrows():
            pairs = []
            for k, v in row.items():
                vs = as_text(v).strip()
                ks = as_text(k).strip()
                if vs and vs.lower() not in {'nan', 'none', '', '-', '--', 'n/a'} and ks:
                    pairs.append(f"{ks}: {vs}")
            if len(pairs) < 2:
                continue
            prefix = f"[{path.name}{'|'+sheet if sheet else ''}] "
            text = prefix + " | ".join(pairs)
            chunks.append({
                'text': text[:700],
                'source': path.name,
                'sheet': sheet or '',
            })
    return chunks


def _embed_batch(texts: list[str], client, batch_size: int = 200) -> np.ndarray:
    all_vecs: list = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model=_EMBED_MODEL)
        all_vecs.extend([e.embedding for e in resp.data])
    return np.array(all_vecs, dtype=np.float32)


class ReferenceRAG:
    """Lightweight RAG over Rail Baltica reference files.

    Embeds every row of every reference Excel/CSV as a text chunk using
    OpenAI text-embedding-3-small.  Retrieval uses cosine similarity — no
    external vector database required.  The index is persisted to disk so
    it is rebuilt only when reference files change.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._ready = False

    # ------------------------------------------------------------------ build

    def build(self, ref_dirs: list, api_key: str, cache_dir=None, status_cb=None):
        if not api_key or not _OPENAI_OK:
            return
        ref_dirs = [Path(d) for d in ref_dirs if Path(d).exists()]
        all_files = [
            p for d in ref_dirs
            for p in sorted(d.rglob('*'))
            if p.suffix.lower() in _TABLE_EXTS and not p.name.startswith('~$')
        ]
        if not all_files:
            return

        file_hash = _file_hash(all_files)
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            emb_file = cache_dir / f'emb_{file_hash}.npy'
            meta_file = cache_dir / f'meta_{file_hash}.json'
            if emb_file.exists() and meta_file.exists():
                try:
                    self._embeddings = np.load(str(emb_file))
                    with open(meta_file, encoding='utf-8') as f:
                        self.chunks = json.load(f)
                    self._ready = True
                    return
                except Exception:
                    pass

        if status_cb:
            status_cb(f'Extracting chunks from {len(all_files)} reference files...')
        self.chunks = []
        for p in all_files:
            self.chunks.extend(_chunks_from_file(p))
        if not self.chunks:
            return

        if status_cb:
            status_cb(f'Embedding {len(self.chunks)} reference chunks (one-time, ~1 min)...')
        client = OpenAI(api_key=api_key)
        self._embeddings = _embed_batch([c['text'] for c in self.chunks], client)
        self._ready = True

        if cache_dir:
            try:
                np.save(str(emb_file), self._embeddings)
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(self.chunks, f, ensure_ascii=False)
            except Exception:
                pass

    # --------------------------------------------------------------- retrieve

    @property
    def ready(self) -> bool:
        return self._ready and self._embeddings is not None and bool(self.chunks)

    def retrieve(self, query: str, api_key: str, k: int = 5) -> list[dict]:
        if not self.ready or not api_key or not query.strip():
            return []
        try:
            client = OpenAI(api_key=api_key)
            q_vec = np.array(
                client.embeddings.create(input=[query[:512]], model=_EMBED_MODEL).data[0].embedding,
                dtype=np.float32,
            )
            norms = np.linalg.norm(self._embeddings, axis=1)
            q_norm = np.linalg.norm(q_vec)
            sims = self._embeddings @ q_vec / (norms * q_norm + 1e-9)
            top_idx = np.argsort(sims)[::-1][:k]
            return [
                {'text': self.chunks[i]['text'], 'source': self.chunks[i]['source'], 'score': float(sims[i])}
                for i in top_idx if sims[i] >= _MIN_SCORE
            ]
        except Exception:
            return []

    def stats(self) -> dict:
        return {
            'rag_ready': self._ready,
            'chunks_indexed': len(self.chunks),
            'reference_files_indexed': len({c['source'] for c in self.chunks}),
        }
