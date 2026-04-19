"""
lightweight_retriever_eval.py
=============================
No-giant-download first-stage retrieval comparison.

Compares:
  A) Baseline retriever from dspy_chatbot_compression.py
  B) Lightweight reretrieval over candidate pool using small embedding models

This avoids large LLM retriever checkpoints (e.g., 30GB+ BGE-Reasoner models).

Install:
  pip install sentence-transformers numpy

Run:
  python lightweight_retriever_eval.py --n 10
  python lightweight_retriever_eval.py --models BAAI/bge-base-en-v1.5 sentence-transformers/all-MiniLM-L6-v2
  python lightweight_retriever_eval.py --pool-k 40 --final-k 5

Outputs:
  lightweight_retriever_results.json
  lightweight_retriever_report.txt
"""

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import dspy_chatbot_compression as bot
from evaluate import (
    EVAL_SET,
    keyword_coverage,
    citation_rate,
    hallucination_flag,
    answer_completeness,
    context_precision,
    mrr,
    avg,
)

rag = bot.rag
retriever = bot.retriever
lm = bot.lm

import dspy

dspy.configure(lm=lm)


DEFAULT_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]


@dataclass
class RunResult:
    model_name: str
    query_id: str
    question: str
    final_rewrite: str
    n_chunks: int
    answer: str
    keyword_coverage: float
    citation_rate: float
    hallucination: bool
    context_precision: float
    mrr_score: float
    completeness: float
    retrieve_ms: float
    generate_ms: float
    total_ms: float
    error: str = ""


@dataclass
class VariantResult:
    query_id: str
    question: str
    category: str
    complexity: str
    baseline: RunResult
    variants: dict


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _load_embedder(model_id: str):
    from sentence_transformers import SentenceTransformer

    print(f"Loading lightweight embedder: {model_id}")
    model = SentenceTransformer(model_id)
    return model


def _reretrieve(encoder, query: str, passages: list[str], top_k: int) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    if not passages:
        return [], 0.0
    q = encoder.encode([query], convert_to_numpy=True)
    p = encoder.encode(passages, convert_to_numpy=True)
    q = _norm(np.asarray(q, dtype=np.float32))
    p = _norm(np.asarray(p, dtype=np.float32))
    scores = (q @ p.T).reshape(-1)
    ranked = [passages[i] for i in np.argsort(-scores)[:top_k]]
    return ranked, (time.perf_counter() - t0) * 1000


def _build_chunks(passages: list[str]):
    class _Chunk:
        def __init__(self, text, idx):
            self.chunk_id = idx + 1
            self.original = text
            self.compressed = text
            self.was_compressed = False

    return [_Chunk(p, i) for i, p in enumerate(passages)]


def _answer(question: str, passages: list[str]) -> tuple[str, list, float]:
    chunks = _build_chunks(passages)
    context = [f"[Chunk {c.chunk_id}] {c.compressed}" for c in chunks if c.compressed.strip()]
    t0 = time.perf_counter()
    pred = rag.generate_answer(context=context, question=question)
    return pred.answer, chunks, (time.perf_counter() - t0) * 1000


def _evaluate_single(item: dict, encoders: dict, pool_k: int, final_k: int) -> VariantResult:
    qid = item["id"]
    question = item["question"]
    keywords = item["expected_keywords"]
    reference = item["reference_answer"]

    rw = rag.rewriter(raw_question=question)
    final_rewrite = rw.final_rewrite

    # Baseline
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    base_passages = retriever(final_rewrite, k=final_k)
    base_ret_ms = (time.perf_counter() - t0) * 1000
    base_answer, base_chunks, base_gen_ms = _answer(question, base_passages)
    base_total_ms = (time.perf_counter() - t_total) * 1000
    baseline = RunResult(
        model_name="Baseline",
        query_id=qid,
        question=question,
        final_rewrite=final_rewrite,
        n_chunks=len(base_chunks),
        answer=base_answer,
        keyword_coverage=keyword_coverage(base_answer, keywords),
        citation_rate=citation_rate(base_answer),
        hallucination=hallucination_flag(base_answer, base_chunks),
        context_precision=context_precision(base_chunks, keywords),
        mrr_score=mrr(base_chunks, keywords),
        completeness=answer_completeness(base_answer, reference),
        retrieve_ms=base_ret_ms,
        generate_ms=base_gen_ms,
        total_ms=base_total_ms,
    )

    variants = {}
    for model_id, encoder in encoders.items():
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        pool = retriever(final_rewrite, k=pool_k)
        selected, rr_ms = _reretrieve(encoder, final_rewrite, pool, final_k)
        ret_ms = (time.perf_counter() - t0) * 1000 + rr_ms
        ans, chunks, gen_ms = _answer(question, selected)
        total_ms = (time.perf_counter() - t_total) * 1000
        variants[model_id] = RunResult(
            model_name=model_id,
            query_id=qid,
            question=question,
            final_rewrite=final_rewrite,
            n_chunks=len(chunks),
            answer=ans,
            keyword_coverage=keyword_coverage(ans, keywords),
            citation_rate=citation_rate(ans),
            hallucination=hallucination_flag(ans, chunks),
            context_precision=context_precision(chunks, keywords),
            mrr_score=mrr(chunks, keywords),
            completeness=answer_completeness(ans, reference),
            retrieve_ms=ret_ms,
            generate_ms=gen_ms,
            total_ms=total_ms,
        )

    return VariantResult(
        query_id=qid,
        question=question,
        category=item.get("category", "?"),
        complexity=item.get("complexity", "SIMPLE"),
        baseline=baseline,
        variants=variants,
    )


def run_all(eval_set: list, encoders: dict, max_queries: int | None, pool_k: int, final_k: int):
    subset = eval_set[:max_queries] if max_queries else eval_set
    results = []
    print(f"\n{'='*74}")
    print(f"  Lightweight first-stage comparison on {len(subset)} queries")
    print(f"  Baseline + {len(encoders)} lightweight reretrieval variants")
    print(f"{'='*74}\n")
    for i, item in enumerate(subset, 1):
        print(f"[{i:>2}/{len(subset)}] {item['id']}: {item['question'][:50]}...")
        try:
            r = _evaluate_single(item, encoders=encoders, pool_k=pool_k, final_k=final_k)
            print(
                f"  Baseline: kc={r.baseline.keyword_coverage:.2f} "
                f"cp={r.baseline.context_precision:.2f} total={r.baseline.total_ms:.0f}ms"
            )
            for model_id, rr in r.variants.items():
                print(
                    f"  {model_id}: kc={rr.keyword_coverage:.2f} "
                    f"cp={rr.context_precision:.2f} total={rr.total_ms:.0f}ms"
                )
            results.append(r)
        except Exception as e:
            print(f"  ❌ {e}")
            empty = RunResult(
                model_name="ERROR",
                query_id=item["id"],
                question=item["question"],
                final_rewrite="",
                n_chunks=0,
                answer="",
                keyword_coverage=0.0,
                citation_rate=0.0,
                hallucination=False,
                context_precision=0.0,
                mrr_score=0.0,
                completeness=0.0,
                retrieve_ms=0.0,
                generate_ms=0.0,
                total_ms=0.0,
                error=str(e),
            )
            results.append(
                VariantResult(
                    query_id=item["id"],
                    question=item["question"],
                    category=item.get("category", "?"),
                    complexity=item.get("complexity", "?"),
                    baseline=empty,
                    variants={k: empty for k in encoders.keys()},
                )
            )
        time.sleep(0.2)
    return results


def _mean_metric(rows: list[VariantResult], model_key: str, attr: str) -> float:
    vals = []
    for r in rows:
        if model_key == "baseline":
            vals.append(getattr(r.baseline, attr))
        else:
            vals.append(getattr(r.variants[model_key], attr))
    return avg(vals)


def generate_report(results: list[VariantResult], model_ids: list[str], pool_k: int, final_k: int) -> str:
    ok = [r for r in results if not r.baseline.error]
    lines = []
    p = lambda *a: lines.append(" ".join(str(x) for x in a))

    p("=" * 86)
    p("  LIGHTWEIGHT RETRIEVER EVAL REPORT (NO GIANT DOWNLOAD)")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p(f"  Setup: candidate pool {pool_k} -> top {final_k}")
    p("=" * 86)
    p(f"  Queries: {len(results)} | OK: {len(ok)}")

    p()
    p("── METRICS (AVERAGE OVER OK QUERIES) ───────────────────────────────────────────")
    p(f"  {'Model':<45} {'KC':>6} {'CP':>6} {'MRR':>6} {'COMP':>7} {'TOT(ms)':>10}")
    p(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")
    p(
        f"  {'Baseline':<45} "
        f"{_mean_metric(ok,'baseline','keyword_coverage'):>6.3f} "
        f"{_mean_metric(ok,'baseline','context_precision'):>6.3f} "
        f"{_mean_metric(ok,'baseline','mrr_score'):>6.3f} "
        f"{_mean_metric(ok,'baseline','completeness'):>7.3f} "
        f"{_mean_metric(ok,'baseline','total_ms'):>10.0f}"
    )
    for model_id in model_ids:
        p(
            f"  {model_id:<45} "
            f"{_mean_metric(ok,model_id,'keyword_coverage'):>6.3f} "
            f"{_mean_metric(ok,model_id,'context_precision'):>6.3f} "
            f"{_mean_metric(ok,model_id,'mrr_score'):>6.3f} "
            f"{_mean_metric(ok,model_id,'completeness'):>7.3f} "
            f"{_mean_metric(ok,model_id,'total_ms'):>10.0f}"
        )

    p()
    p("── DELTA VS BASELINE ────────────────────────────────────────────────────────────")
    p(f"  {'Model':<45} {'ΔKC':>8} {'ΔCP':>8} {'ΔMRR':>8} {'ΔCOMP':>8} {'ΔTOT(ms)':>10}")
    for model_id in model_ids:
        p(
            f"  {model_id:<45} "
            f"{_mean_metric(ok,model_id,'keyword_coverage') - _mean_metric(ok,'baseline','keyword_coverage'):>+8.3f} "
            f"{_mean_metric(ok,model_id,'context_precision') - _mean_metric(ok,'baseline','context_precision'):>+8.3f} "
            f"{_mean_metric(ok,model_id,'mrr_score') - _mean_metric(ok,'baseline','mrr_score'):>+8.3f} "
            f"{_mean_metric(ok,model_id,'completeness') - _mean_metric(ok,'baseline','completeness'):>+8.3f} "
            f"{_mean_metric(ok,model_id,'total_ms') - _mean_metric(ok,'baseline','total_ms'):>+10.0f}"
        )

    p()
    p("Note: These variants rerank a baseline candidate pool; they do not require")
    p("rebuilding your vector DB and avoid very large model downloads.")
    p("=" * 86)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate lightweight first-stage reretrieval variants")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N queries")
    parser.add_argument("--pool-k", type=int, default=40, help="Candidate pool size from baseline retriever")
    parser.add_argument("--final-k", type=int, default=5, help="Final top-k passages")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="SentenceTransformer model IDs")
    parser.add_argument("--out", default="lightweight_retriever_results.json")
    parser.add_argument("--report", default="lightweight_retriever_report.txt")
    args = parser.parse_args()

    encoders = {m: _load_embedder(m) for m in args.models}
    results = run_all(EVAL_SET, encoders=encoders, max_queries=args.n, pool_k=args.pool_k, final_k=args.final_k)

    serial = []
    for r in results:
        serial.append(
            {
                "query_id": r.query_id,
                "question": r.question,
                "category": r.category,
                "complexity": r.complexity,
                "baseline": asdict(r.baseline),
                "variants": {k: asdict(v) for k, v in r.variants.items()},
            }
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)
    print(f"\n✅  JSON   → {args.out}")

    report = generate_report(results, model_ids=args.models, pool_k=args.pool_k, final_k=args.final_k)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅  Report → {args.report}\n")
    print(report)

