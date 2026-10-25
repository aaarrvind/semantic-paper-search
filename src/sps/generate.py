"""Optional answer layer: a small open LLM writes a short answer that cites the retrieved abstracts.

Grounding rules (also the main ethical safeguard):
- the model only sees the retrieved passages and is told to use nothing else;
- every sentence must cite a passage number like [2] (one worked example shows the format);
- an answer that is only citations (e.g. "[1]") is regenerated once with a sentence started for the model;
- if retrieval confidence is low, no answer is generated at all (passages are shown instead).
"""

import re

NO_EVIDENCE = "Not enough evidence in the retrieved papers to answer this."

SYSTEM_PROMPT = (
    "You help researchers by summarising what scientific passages say. "
    "The user gives a scientific claim or a question, followed by numbered passages. "
    "Write 2-3 complete sentences in your own words saying what the passages report about it: "
    "whether they support it, contradict it, or only partly address it, and the key finding. "
    "End every sentence with the passage number(s) it is based on, like [1] or [2][3]. "
    "Never reply with only a citation. "
    f"If no passage addresses the claim or question, reply exactly: \"{NO_EVIDENCE}\" "
    "Use ONLY the passages: no outside knowledge and no invented citations."
)

# One worked example (made up, not from SciFact) so a small model sees the expected format.
_EXAMPLE_USER = (
    "Passages:\n[1] Daily walking and blood pressure. In a 12-week trial, 80 adults who walked 30 minutes "
    "a day lowered their systolic blood pressure by 6 mmHg on average compared with controls.\n\n"
    "[2] Coffee intake and sleep. Drinking coffee after 4 pm reduced total sleep time in young adults.\n\n"
    "Claim or question: Regular walking reduces blood pressure."
)
_EXAMPLE_ANSWER = (
    "Passage [1] supports this claim: in a 12-week trial, adults who walked 30 minutes a day lowered their "
    "systolic blood pressure by about 6 mmHg compared with controls [1]. "
    "Passage [2] is about coffee and sleep and does not address blood pressure [2]."
)

# Used only on a retry, when the first output was nothing but citations.
_RETRY_PREFIX = "According to the passages,"


def build_prompt(question, passages):
    """passages: list of (title, text). Returns chat messages."""
    context = "\n\n".join(f"[{i}] {title}. {text[:1200]}" for i, (title, text) in enumerate(passages, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _EXAMPLE_USER},
        {"role": "assistant", "content": _EXAMPLE_ANSWER},
        {"role": "user", "content": f"Passages:\n{context}\n\nClaim or question: {question}"},
    ]


def is_citation_only(text):
    """True when the output has (almost) no words apart from citation markers like [1]."""
    words = re.sub(r"\[\d+\]", " ", text)
    return len(re.findall(r"[A-Za-z]{2,}", words)) < 4


def _new_dtype_arg():
    """transformers >= 4.56 renamed `torch_dtype` to `dtype`."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return (major, minor) >= (4, 56)


class CitedAnswerGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype) if _new_dtype_arg() \
            else AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def _generate(self, messages, prefix=""):
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        n_prompt = inputs["input_ids"].shape[1]
        return (prefix + self.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)).strip()

    def answer(self, question, passages):
        messages = build_prompt(question, passages)
        text = self._generate(messages)
        if text.startswith(NO_EVIDENCE[:20]) or not is_citation_only(text):
            return text
        # Small models sometimes answer a claim with a bare "[1]": start the sentence for them and retry once.
        return self._generate(messages, prefix=_RETRY_PREFIX)


def answer_with_guardrail(generator, question, results, corpus, min_score=None, top_n=3):
    """results: [(doc_index, score), ...] from a dense retriever (cosine scores).
    Skips generation when the best score is below min_score."""
    if not results or (min_score is not None and results[0][1] < min_score):
        return NO_EVIDENCE, []
    used = results[:top_n]
    passages = [(corpus.titles[i], corpus.texts[i]) for i, _ in used]
    return generator.answer(question, passages), [i for i, _ in used]
