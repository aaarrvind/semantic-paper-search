"""Retrieval models compared in the project.

Every retriever exposes the same method:
    search(query, k) -> list of (doc_index, score), best first
so they can be evaluated, fused and shown in the demo interchangeably.
"""

import time

import numpy as np

from .preprocess import BM25Preprocessor


class BM25Retriever:
    """Keyword baseline: Okapi BM25 (rank_bm25) over pre-processed tokens."""

    def __init__(self, texts, preprocessor=None, k1=1.5, b=0.75):
        from rank_bm25 import BM25Okapi

        self.pre = preprocessor or BM25Preprocessor()
        self.name = f"BM25 ({self.pre.describe()})"
        self.short = "BM25"
        t0 = time.time()
        self.bm25 = BM25Okapi([self.pre(t) for t in texts], k1=k1, b=b)
        self.index_seconds = time.time() - t0

    def scores(self, query):
        return self.bm25.get_scores(self.pre(query))

    def search(self, query, k=10):
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]


class DenseRetriever:
    """Dense retrieval: sentence-transformer embeddings + cosine similarity (FAISS inner-product index).

    query_prefix: some models (e.g. BGE) expect an instruction before short queries.
    """

    def __init__(self, texts, model_name="sentence-transformers/all-MiniLM-L6-v2", query_prefix="",
                 batch_size=64, device=None, embeddings=None, model=None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.short = model_name.split("/")[-1]
        self.name = f"Dense ({self.short}{', with query prefix' if query_prefix else ''})"
        self.model = model or SentenceTransformer(model_name, device=device)
        t0 = time.time()
        if embeddings is None:
            embeddings = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                           show_progress_bar=True, convert_to_numpy=True)
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.index_seconds = time.time() - t0
        self._build_index()

    def _build_index(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except ImportError:          # plain numpy works fine for a few thousand documents
            self.index = None

    def encode_query(self, query):
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    def search(self, query, k=10):
        q = self.encode_query(query)
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        s = self.embeddings @ q[0]
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]

    def save_embeddings(self, path):
        np.save(path, self.embeddings)


class HybridRetriever:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over retrievers of 1 / (rrf_k + rank(d)).

    Uses ranks only, so BM25 scores and cosine similarities need no normalisation. rrf_k=60 is the standard value.
    """

    def __init__(self, retrievers, rrf_k=60, depth=100):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth
        self.short = "Hybrid"
        self.name = "Hybrid RRF (" + " + ".join(r.short for r in retrievers) + ")"

    def search(self, query, k=10):
        fused = {}
        for r in self.retrievers:
            for rank, (doc, _) in enumerate(r.search(query, self.depth), start=1):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


class CrossEncoderReranker:
    """Optional second stage: a cross-encoder reads (query, document) pairs together and re-scores the top candidates."""

    def __init__(self, first_stage, texts, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", candidates=20, device=None):
        from sentence_transformers import CrossEncoder

        self.first_stage = first_stage
        self.texts = texts
        self.candidates = candidates
        self.model = CrossEncoder(model_name, device=device)
        self.short = first_stage.short + "+rerank"
        self.name = f"{first_stage.name} + re-ranker"

    def search(self, query, k=10):
        cands = self.first_stage.search(query, max(k, self.candidates))
        scores = self.model.predict([(query, self.texts[d]) for d, _ in cands])
        ranked = sorted(zip([d for d, _ in cands], scores), key=lambda x: -x[1])
        return [(int(d), float(s)) for d, s in ranked[:k]]
