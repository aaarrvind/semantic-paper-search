"""Loading the document collection, test queries and relevance labels."""

import re
from dataclasses import dataclass, field


def clean_text(text):
    """Collapse whitespace and line breaks; keep the original wording otherwise."""
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class Corpus:
    """A document collection plus (optionally) labelled test queries.

    doc_ids[i], titles[i], texts[i] describe document i.
    queries: {query_id: query_text}
    qrels:   {query_id: {doc_id: relevance}}  (relevance > 0 means relevant)
    """
    doc_ids: list
    titles: list
    texts: list
    urls: list = field(default_factory=list)
    queries: dict = field(default_factory=dict)
    qrels: dict = field(default_factory=dict)
    name: str = "corpus"

    def __len__(self):
        return len(self.doc_ids)

    def full_text(self, i):
        """Title and abstract joined, as used for indexing."""
        title = self.titles[i]
        return f"{title.rstrip('.')}. {self.texts[i]}" if title else self.texts[i]

    def all_full_texts(self):
        return [self.full_text(i) for i in range(len(self))]

    def index_of(self):
        return {d: i for i, d in enumerate(self.doc_ids)}

    def stats(self):
        words = [len(self.full_text(i).split()) for i in range(len(self))]
        q_words = [len(q.split()) for q in self.queries.values()]
        rel = [sum(1 for r in v.values() if r > 0) for v in self.qrels.values()]
        return {
            "documents": len(self),
            "avg words per document": round(sum(words) / max(len(words), 1), 1),
            "test queries": len(self.queries),
            "avg words per query": round(sum(q_words) / max(len(q_words), 1), 1),
            "avg relevant docs per query": round(sum(rel) / max(len(rel), 1), 2),
        }


def load_scifact(split="test"):
    """BEIR SciFact from the Hugging Face Hub: 5,183 abstracts, 300 labelled test claims.

    Sources: BeIR/scifact (configs 'corpus' and 'queries') and BeIR/scifact-qrels.
    Only queries that have relevance labels in the chosen split are kept.
    """
    from datasets import load_dataset

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_rows = load_dataset("BeIR/scifact-qrels", split=split)

    qrels = {}
    for row in qrels_rows:
        qid, did, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        qrels.setdefault(qid, {})[did] = score
    query_text = {str(r["_id"]): clean_text(r["text"]) for r in queries}

    return Corpus(
        doc_ids=[str(r["_id"]) for r in corpus],
        titles=[clean_text(r["title"]) for r in corpus],
        texts=[clean_text(r["text"]) for r in corpus],
        urls=[f"https://www.semanticscholar.org/search?q={clean_text(r['title']).replace(' ', '+')}" for r in corpus],
        queries={q: query_text[q] for q in qrels if q in query_text},
        qrels=qrels,
        name=f"scifact-{split}",
    )


def load_custom_corpus(records, name="custom"):
    """Build an unlabelled corpus (for the demo) from dicts with 'title', 'text' and optional 'id', 'url'."""
    return Corpus(
        doc_ids=[str(r.get("id", i)) for i, r in enumerate(records)],
        titles=[clean_text(r.get("title", "")) for r in records],
        texts=[clean_text(r.get("text", "")) for r in records],
        urls=[r.get("url", "") for r in records],
        name=name,
    )
