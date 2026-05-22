from __future__ import annotations
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from .io_utils import as_text

_TABLE_EXTS = {'.xlsx', '.xlsm', '.xls', '.csv'}
_OPENAI_EMBED_MODEL = 'text-embedding-3-small'
_GOOGLE_EMBED_MODEL = 'models/text-embedding-004'
_MIN_SCORE = 0.28


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_openai(texts: list[str], api_key: str, batch_size: int = 200) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    all_vecs: list = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model=_OPENAI_EMBED_MODEL)
        all_vecs.extend([e.embedding for e in resp.data])
    return np.array(all_vecs, dtype=np.float32)


def _embed_google(texts: list[str], api_key: str, batch_size: int = 100,
                  task_type: str = 'retrieval_document') -> np.ndarray:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    all_vecs: list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = genai.embed_content(model=_GOOGLE_EMBED_MODEL, content=batch, task_type=task_type)
        embs = result['embedding']
        if embs and isinstance(embs[0], list):
            all_vecs.extend(embs)
        else:
            all_vecs.append(embs)
    return np.array(all_vecs, dtype=np.float32)


def _embed_query(query: str, api_key: str, provider: str) -> np.ndarray:
    if provider == 'google':
        return _embed_google([query], api_key, task_type='retrieval_query')[0]
    else:
        return _embed_openai([query], api_key)[0]


# ── Chunk extraction ──────────────────────────────────────────────────────────

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
            pairs = [
                f"{as_text(k).strip()}: {as_text(v).strip()}"
                for k, v in row.items()
                if as_text(v).strip() and as_text(v).strip().lower() not in {'nan', 'none', '-', '--', 'n/a'}
                and as_text(k).strip()
            ]
            if len(pairs) < 2:
                continue
            prefix = f"[{path.name}{'|'+sheet if sheet else ''}] "
            chunks.append({
                'text': (prefix + ' | '.join(pairs))[:700],
                'source': path.name,
                'sheet': sheet or '',
            })
    return chunks


# ── RAG class ─────────────────────────────────────────────────────────────────

class ReferenceRAG:
    """Lightweight RAG over Rail Baltica reference files.

    Supports both OpenAI (text-embedding-3-small) and Google Gemini
    (text-embedding-004, free tier) for embeddings.  No external vector
    database — pure cosine similarity over a numpy matrix.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._ready = False
        self._provider = 'openai'
        self._api_key = ''

    def build(self, ref_dirs: list, api_key: str, provider: str = 'openai',
              cache_dir=None, status_cb=None):
        if not api_key:
            return
        self._provider = provider.lower()
        self._api_key = api_key
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
            emb_file = cache_dir / f'emb_{self._provider}_{file_hash}.npy'
            meta_file = cache_dir / f'meta_{self._provider}_{file_hash}.json'
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
            status_cb(f'Embedding {len(self.chunks)} reference chunks via {provider} (one-time, ~1 min)...')
        texts = [c['text'] for c in self.chunks]
        try:
            if self._provider == 'google':
                self._embeddings = _embed_google(texts, api_key)
            else:
                self._embeddings = _embed_openai(texts, api_key)
        except Exception:
            return
        self._ready = True

        if cache_dir:
            try:
                np.save(str(emb_file), self._embeddings)
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(self.chunks, f, ensure_ascii=False)
            except Exception:
                pass

    @property
    def ready(self) -> bool:
        return self._ready and self._embeddings is not None and bool(self.chunks)

    def retrieve(self, query: str, k: int = 6) -> list[dict]:
        if not self.ready or not query.strip():
            return []
        try:
            q_vec = _embed_query(query[:512], self._api_key, self._provider)
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
            'provider': self._provider,
            'chunks_indexed': len(self.chunks),
            'reference_files_indexed': len({c['source'] for c in self.chunks}),
        }
