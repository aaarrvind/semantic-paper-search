# Semantic Paper Search

**Keyword vs dense vs hybrid retrieval for scientific literature.** MAAI 902 (NLP & LLMs), Task 2: NLP Solution Design.

Type a natural-language question or claim and get the most relevant research abstracts, ranked, with their sources. The project compares three ways of finding them:

| Method | Idea | Strength | Weakness |
|---|---|---|---|
| **BM25** (keyword) | Scores documents by shared terms, weighted by rarity | Exact technical terms, names, numbers | Misses synonyms and paraphrases |
| **Dense** (semantic) | Embeds query and documents as vectors; ranks by cosine similarity | Matches meaning, not just words | Can miss rare exact terms; needs a model |
| **Hybrid** (RRF) | Fuses both rankings with Reciprocal Rank Fusion | Combines both strengths | Slightly slower |

An **optional** answer layer has a small open LLM write a short answer from the top abstracts, citing each one, or refuse when the evidence is weak.

## Results

Run on Google Colab (T4 GPU), BEIR SciFact test set (300 claims). Full table: `results/all_results.csv`.

| Method | nDCG@10 | Recall@100 | MRR@10 | ms / query |
|---|---|---|---|---|
| BM25 | 0.670 | 0.910 | 0.632 | 9.7 |
| Dense: all-MiniLM-L6-v2 | 0.650 | 0.925 | 0.610 | 7.7 |
| Dense: bge-small-en-v1.5 | 0.710 | 0.943 | 0.678 | 11.8 |
| **Hybrid RRF** | **0.721** | **0.972** | **0.690** | 22.7 |
| Hybrid RRF + re-ranker | 0.688 | 0.972 | 0.659 | 562.4 |

## Data
[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) (Wadden et al., 2020; Thakur et al., 2021): 5,183 scientific abstracts, 300 expert-written test claims, and relevance labels ([BeIR/scifact-qrels](https://huggingface.co/datasets/BeIR/scifact-qrels)). Everything is downloaded automatically from Hugging Face; no data is stored in this repo.

## Project structure
```
├── app.py                      Gradio demo (search, side-by-side comparison, optional cited answers)
├── notebooks/experiments.ipynb All experiments, tables, charts and error analysis (run in Colab)
├── requirements.txt
└── src/sps/
    ├── data.py                 Loads the corpus, queries and relevance labels
    ├── preprocess.py           BM25 pre-processing (lowercase, stop-words, Porter stemming)
    ├── retrievers.py           BM25, Dense (sentence-transformers + FAISS), Hybrid (RRF), cross-encoder re-ranker
    ├── evaluate.py             nDCG@k, Recall@k, Precision@k, MRR@10
    └── generate.py             Optional grounded answer generation with citations and a refusal guardrail
```

## How to run

**Experiments (Google Colab, T4 GPU):** open `notebooks/experiments.ipynb` in Colab, set `REPO_URL` in the first cell to this repository, then **Runtime → Run all**.

**Demo (locally or in Colab):**
```bash
pip install -r requirements.txt
python app.py              # retrieval only
python app.py --answers    # + cited answers (GPU recommended)
python app.py --share      # public link, e.g. from Colab
```

## Models and tools
`rank_bm25` (BM25) · `sentence-transformers` with `all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` (embeddings) · FAISS (vector index) · `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional re-ranker) · `Qwen/Qwen2.5-1.5B-Instruct` (optional answers) · Hugging Face `datasets` · Gradio (UI).

## Responsible use
Results come from a biomedical abstract collection and may be incomplete or biased towards well-studied topics. Generated answers can still be wrong, so always read the cited papers. Only abstracts are used (no full texts), and queries are not logged.

## References
- Thakur, N. et al. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. NeurIPS Datasets & Benchmarks.
- Wadden, D. et al. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP 2020.
- Zhao, W. X., Liu, J., Ren, R., Wen, J.-R. (2024). Dense text retrieval based on pretrained language models: A survey. ACM TOIS. https://doi.org/10.1145/3637870
- Zhu, Y. et al. (2025). Large language models for information retrieval: A survey. ACM TOIS. https://doi.org/10.1145/3748304
- Ni, B. et al. (2026). Towards trustworthy retrieval augmented generation for large language models: A survey. ACM Computing Surveys. https://doi.org/10.1145/3837074
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR 2009.
