"""Gradio demo: type a question, compare keyword / semantic / hybrid search, optionally get a cited answer.

Run in Colab or locally:
    python app.py                 # SciFact corpus, retrieval only
    python app.py --answers       # also load the small LLM for cited answers (GPU recommended)
    python app.py --share         # public link (needed when running inside Colab)
"""

import argparse
import html
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import gradio as gr

from sps import load_scifact, BM25Retriever, DenseRetriever, HybridRetriever

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE = "cache/bge-small-en-v1.5.npy"   # same file the notebook creates
MIN_ANSWER_SCORE = 0.55      # dense cosine below this -> no generated answer (tune on a few queries)

EXAMPLES = [
    "Does vitamin D supplementation reduce the risk of fractures?",
    "What role does microRNA play in cancer progression?",
    "Is obesity linked to a higher risk of type 2 diabetes?",
]


def build(answers=False):
    corpus = load_scifact("test")
    texts = corpus.all_full_texts()
    bm25 = BM25Retriever(texts)
    emb = np.load(CACHE) if os.path.exists(CACHE) else None
    dense = DenseRetriever(texts, DENSE_MODEL, query_prefix=BGE_PREFIX, embeddings=emb)
    if emb is None:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        dense.save_embeddings(CACHE)
    hybrid = HybridRetriever([bm25, dense])
    gen = None
    if answers:
        from sps.generate import CitedAnswerGenerator
        gen = CitedAnswerGenerator()
    return corpus, {"Keyword (BM25)": bm25, "Semantic (dense)": dense, "Hybrid (RRF)": hybrid}, dense, gen


def render(results, corpus, query):
    terms = [re.escape(t) for t in re.findall(r"[A-Za-z0-9-]{4,}", query)]
    pattern = re.compile(r"\b(" + "|".join(terms) + r")\b", re.IGNORECASE) if terms else None
    cards = []
    for rank, (i, score) in enumerate(results, start=1):
        snippet = html.escape(corpus.texts[i][:420]) + ("..." if len(corpus.texts[i]) > 420 else "")
        if pattern:   # light highlight of query words
            snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)
        link = f' · <a href="{corpus.urls[i]}" target="_blank">find paper</a>' if corpus.urls else ""
        cards.append(
            f"<div style='margin:0 0 14px'><b>{rank}. {html.escape(corpus.titles[i])}</b><br>"
            f"<small>doc {corpus.doc_ids[i]} · score {score:.3f}{link}</small><br>{snippet}</div>")
    return "".join(cards) or "<i>No results</i>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="load the LLM for cited answers")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()

    corpus, methods, dense, gen = build(args.answers)

    def search(query, method, k, compare):
        if not query.strip():
            return "", "", ""
        left = render(methods[method].search(query, int(k)), corpus, query)
        right = render(methods["Keyword (BM25)"].search(query, int(k)), corpus, query) if compare else ""
        return left, right, ""

    def answer(query):
        if gen is None:
            return "Answer generation is off. Restart with `python app.py --answers`."
        from sps.generate import answer_with_guardrail
        text, used = answer_with_guardrail(gen, query, dense.search(query, 5), corpus, MIN_ANSWER_SCORE)
        sources = "\n".join(f"[{n}] {corpus.titles[i]}" for n, i in enumerate(used, start=1))
        return f"{text}\n\nSources:\n{sources}" if used else text

    with gr.Blocks(title="Semantic Paper Search") as demo:
        gr.Markdown("# Semantic Paper Search\nSearch 5,183 scientific abstracts (SciFact) by meaning, by keyword, or both. "
                    "Answers, if enabled, are generated only from the retrieved abstracts and cite them.")
        with gr.Row():
            q = gr.Textbox(label="Question or claim", placeholder=EXAMPLES[0], scale=4)
            method = gr.Dropdown(list(methods), value="Hybrid (RRF)", label="Method")
            k = gr.Slider(1, 20, value=5, step=1, label="Results")
        compare = gr.Checkbox(label="Compare side by side with keyword search (BM25)")
        with gr.Row():
            go = gr.Button("Search", variant="primary")
            ans_btn = gr.Button("Generate cited answer")
        ans = gr.Textbox(label="Answer (from retrieved abstracts only)", lines=5)
        with gr.Row():
            out = gr.HTML()
            out_bm25 = gr.HTML()
        gr.Examples(EXAMPLES, inputs=q)
        go.click(search, [q, method, k, compare], [out, out_bm25, ans])
        q.submit(search, [q, method, k, compare], [out, out_bm25, ans])
        ans_btn.click(answer, q, ans)
        gr.Markdown("<small>Limitations: biomedical abstracts only; results can be incomplete or biased towards well-studied topics; "
                    "always read the source papers.</small>")
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
