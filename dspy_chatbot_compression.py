"""
dspy_chatbot.py
===============
Simple DSPy RAG chatbot over the DKU Bulletin using HuggingFace Inference API.

How dspy.LM connects to HuggingFace (from official DSPy docs):
  "If your provider offers an OpenAI-compatible endpoint,
   just add an openai/ prefix to your full model name."

  lm = dspy.LM(
      model    = "openai/<model-name>",   # openai/ prefix = OpenAI-compatible
      api_base = "https://router.huggingface.co/v1",
      api_key  = HF_TOKEN,
  )

This is the same HuggingFace router URL used in query.py — just wired
through dspy.LM instead of the openai Python client directly.

Install:
  pip install dspy llama-index llama-index-vector-stores-supabase
               llama-index-embeddings-huggingface vecs python-dotenv
"""

import os
import sys
import time
from dataclasses import dataclass
from dotenv import load_dotenv

import dspy
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy.exc import OperationalError


# ── 1. Load credentials ───────────────────────────────────────────────────────
load_dotenv()

DB_URL   = os.getenv("SUPABASE_DB_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

if not DB_URL:
    print("ERROR: SUPABASE_DB_URL not found in .env"); sys.exit(1)
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env"); sys.exit(1)


# ── 2. Configure dspy.LM with HuggingFace Inference API ──────────────────────
#
#   DSPy uses LiteLLM under the hood for all LM calls.
#   LiteLLM supports any OpenAI-compatible endpoint via the "openai/" prefix.
#
#   This is exactly the same as query.py's:
#     OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
#     model="meta-llama/Llama-3.1-8B-Instruct:novita"
#
#   — just expressed as dspy.LM instead.
#
lm = dspy.LM(
    model    = "openai/meta-llama/Llama-3.1-8B-Instruct:novita",
    api_base = "https://router.huggingface.co/v1",
    api_key  = HF_TOKEN,
    max_tokens  = 2048,   # was 512 — increased to avoid answer truncation
    temperature = 0.2,
)
dspy.configure(lm=lm)
print("✅  dspy.LM configured → Llama-3.1-8B via HuggingFace router")


# ── 3. Load embedding model + reconnect to Supabase ───────────────────────────
#
#   LlamaIndex handles embeddings + Supabase retrieval exactly as in query.py.
#   DSPy only takes over for the generation step.
#
print("Loading embedding model and connecting to Supabase...")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None  # LlamaIndex LLM disabled — dspy.LM handles generation

print("✅  Supabase index reloaded. No re-embedding performed.\n")


# ── 4. DSPy-compatible retriever wrapping LlamaIndex + Supabase ───────────────
#
#   ROOT CAUSE of the error:
#     Supabase (and most Postgres hosts) close idle connections after a timeout.
#     The vecs/SQLAlchemy connection pool holds a stale connection, and the next
#     query fails with "server closed the connection unexpectedly".
#
#   FIX:
#     Catch OperationalError, rebuild the entire VectorStore + retriever from
#     scratch (forces a fresh TCP connection), then retry the query once.
#     This is the safest approach because vecs doesn't expose pool.reset().
#
def _build_retriever(db_url: str, k: int):
    """Create a fresh LlamaIndex retriever with a new Supabase connection."""
    vs = SupabaseVectorStore(
        postgres_connection_string=db_url,
        collection_name="dku_bulletin",
        dimension=384,
    )
    idx = VectorStoreIndex.from_vector_store(vs)
    return idx.as_retriever(similarity_top_k=k)


class SupabaseRM:
    """
    Wraps the LlamaIndex + Supabase retriever for DSPy.
    Auto-reconnects if Supabase drops the idle connection.
    """
    def __init__(self, db_url: str, k: int = 5):
        self.db_url = db_url
        self.k = k
        self._retriever = _build_retriever(db_url, k)

    def _reconnect(self):
        print("\n[Supabase] Connection lost — reconnecting...", flush=True)
        time.sleep(1)
        self._retriever = _build_retriever(self.db_url, self.k)
        print("[Supabase] Reconnected.\n", flush=True)

    def __call__(self, query: str, k: int = None) -> list[str]:
        self._retriever.similarity_top_k = k or self.k
        try:
            nodes = self._retriever.retrieve(query)
        except (OperationalError, Exception) as e:
            # Catch stale-connection errors and retry once with a fresh connection
            if "server closed the connection" in str(e) or "OperationalError" in type(e).__name__:
                self._reconnect()
                nodes = self._retriever.retrieve(query)  # retry
            else:
                raise
        return [node.text for node in nodes]


retriever = SupabaseRM(db_url=DB_URL, k=5)


# ── 5. Signatures ─────────────────────────────────────────────────────────────

class RewriteQuery(dspy.Signature):
    """
    You are a search query optimizer for a university bulletin database.
    Given a raw student question (which may contain typos, abbreviations, or
    vague phrasing), produce 3 distinct rewritten search queries that would
    retrieve the most relevant passages.

    Rules:
    - Expand abbreviations (e.g. "appl maths" → "Applied Mathematics", "CS" → "Computer Science")
    - Fix obvious typos
    - Each candidate should approach the topic from a slightly different angle
    - Keep queries concise and factual — no filler words
    """
    raw_question:        str       = dspy.InputField(desc="The student's original, unedited question")
    candidate_rewrites:  list[str] = dspy.OutputField(desc="Exactly 3 rewritten search query candidates")
    final_rewrite:       str       = dspy.OutputField(desc="The single best rewrite to use for retrieval")


class CompressChunk(dspy.Signature):
    """
    You are a precise document compressor.
    Given one passage from a university bulletin and a student's question,
    extract ONLY the sentences or phrases that are directly relevant to
    answering the question. Discard everything else.

    Rules:
    - Keep original wording — do not paraphrase
    - Preserve numbers, course codes, credit counts, and policy names exactly
    - If nothing in the passage is relevant, return an empty string
    - Output must be shorter than the input
    """
    question:            str = dspy.InputField(desc="The student's question")
    passage:             str = dspy.InputField(desc="One retrieved passage from the DKU Bulletin")
    compressed_passage:  str = dspy.OutputField(desc="Only the sentences relevant to the question, verbatim")


class AnswerFromBulletin(dspy.Signature):
    """
    Answer student questions about the DKU Bulletin accurately and concisely.
    Base your answer only on the provided context.
    If the answer is not in the context, say so clearly.

    IMPORTANT: Every factual claim in your answer MUST be followed by a citation
    in the format [Chunk N] referring to the chunk it came from.
    Example: "Students must complete 128 credits [Chunk 1] including the Common Core [Chunk 3]."
    """
    context:  list[str] = dspy.InputField(desc="Relevant passages from the DKU Bulletin, each labelled [Chunk N]")
    question: str        = dspy.InputField(desc="The student's original question")
    answer:   str        = dspy.OutputField(desc="Answer with inline [Chunk N] citations for every factual claim")


# ── 6. Chunk traceability dataclass ───────────────────────────────────────────
@dataclass
class TracedChunk:
    """
    Full audit trail for one retrieved chunk:
      chunk_id      → matches the [Chunk N] label the LLM sees and cites
      original      → verbatim text from Supabase (ground truth)
      compressed    → what was actually fed to the answer LLM
      was_compressed → whether compression changed the text
    """
    chunk_id:       int   # 1-based, matches [Chunk N] labels in the answer
    original:       str   # full verbatim text from Supabase
    compressed:     str   # text passed to the answer LLM (may equal original)
    was_compressed: bool  # True if compression removed any content


# ── 7. Query Rewriter Module ──────────────────────────────────────────────────
class QueryRewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteQuery)

    def forward(self, raw_question: str) -> dspy.Prediction:
        return self.rewrite(raw_question=raw_question)


# ── 8. Prompt Compressor Module ───────────────────────────────────────────────
#
#   Strategy:
#     1. Count total context tokens (rough estimate: chars / 4)
#     2. If total is under CONTEXT_BUDGET, pass chunks through unchanged
#     3. If over budget, compress each chunk individually via LLM,
#        keeping original text for traceability in the log
#
#   Why per-chunk compression (not one big summary)?
#     - Preserves traceability: each compressed piece maps to one source chunk
#     - Avoids the LLM merging unrelated facts from different chunks
#     - Allows partial compression (only long chunks get compressed)
#
CONTEXT_BUDGET = 3000   # ~750 tokens — safe for Llama-3.1-8B's 8k context
                        # leaving room for the prompt template + answer

class PromptCompressor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.compress = dspy.Predict(CompressChunk)

    def forward(self, question: str, chunks: list[str]) -> list[TracedChunk]:
        total_chars = sum(len(c) for c in chunks)

        # ── Fast path: context fits, no compression needed ────────────────────
        if total_chars <= CONTEXT_BUDGET:
            return [
                TracedChunk(
                    chunk_id       = i + 1,
                    original       = chunk,
                    compressed     = chunk,
                    was_compressed = False,
                )
                for i, chunk in enumerate(chunks)
            ]

        # ── Slow path: compress each chunk individually ────────────────────────
        traced = []
        for i, chunk in enumerate(chunks):
            result = self.compress(question=question, passage=chunk)
            compressed = result.compressed_passage.strip()

            # Safety: if LLM returns empty or longer text, keep original
            if not compressed or len(compressed) >= len(chunk):
                compressed     = chunk
                was_compressed = False
            else:
                was_compressed = True

            traced.append(TracedChunk(
                chunk_id       = i + 1,
                original       = chunk,
                compressed     = compressed,
                was_compressed = was_compressed,
            ))

        return traced


# ── 9. RAG Module: Rewrite → Retrieve → Compress → Answer ────────────────────
#
#   Full pipeline per question:
#     LLM #1  QueryRewriter   → clean retrieval query
#     Supabase                → top-5 raw chunks
#     LLM #2…N PromptCompressor → compress each chunk if context too long
#     LLM final ChainOfThought → grounded answer from compressed context
#
class DKUBulletinRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewriter        = QueryRewriter()
        self.compressor      = PromptCompressor()
        self.generate_answer = dspy.ChainOfThought(AnswerFromBulletin)

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1: rewrite
        rewrite_result = self.rewriter(raw_question=question)
        candidates     = rewrite_result.candidate_rewrites
        final_rewrite  = rewrite_result.final_rewrite

        # Step 2: retrieve raw chunks from Supabase
        raw_chunks = retriever(final_rewrite)

        # Step 3: compress if needed, with full traceability
        traced_chunks = self.compressor(question=question, chunks=raw_chunks)

        # Step 4: build the context list passed to the LLM
        #         use compressed text but label each with its chunk ID
        context_for_llm = [
            f"[Chunk {tc.chunk_id}] {tc.compressed}"
            for tc in traced_chunks
            if tc.compressed.strip()   # skip chunks the LLM found irrelevant
        ]

        # Step 5: generate the final answer
        prediction = self.generate_answer(
            context=context_for_llm,
            question=question,
        )

        return dspy.Prediction(
            original_question = question,
            candidates        = candidates,
            final_rewrite     = final_rewrite,
            traced_chunks     = traced_chunks,
            answer            = prediction.answer,
        )


# ── 10. Instantiate ───────────────────────────────────────────────────────────
rag = DKUBulletinRAG()


# ── 11. Loggers ───────────────────────────────────────────────────────────────
W = 62  # box width

def _box_line(text: str, width: int = W) -> str:
    return f"│  {text:<{width - 4}}│"

def log_rewrite(original: str, candidates: list[str], final: str) -> None:
    print("\n┌" + "─" * W + "┐")
    print(_box_line("🔍 QUERY REWRITE LOG"))
    print("├" + "─" * W + "┤")
    orig_display = original if len(original) <= W - 16 else original[:W - 19] + "..."
    print(_box_line(f"Original : {orig_display}"))
    print("├" + "─" * W + "┤")
    print(_box_line("Candidates:"))
    for i, c in enumerate(candidates, 1):
        display = c if len(c) <= W - 10 else c[:W - 13] + "..."
        print(_box_line(f"  {i}. {display}"))
    print("├" + "─" * W + "┤")
    final_display = final if len(final) <= W - 14 else final[:W - 17] + "..."
    print(_box_line(f"✅ Used   : {final_display}"))
    print("└" + "─" * W + "┘")


def _wrap(text: str, width: int = W - 6, indent: str = "    ") -> list[str]:
    """Word-wrap a string into box-width lines."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return [indent + l for l in lines] if lines else [indent + text[:width]]


def log_compression(traced_chunks: list[TracedChunk]) -> None:
    any_compressed = any(tc.was_compressed for tc in traced_chunks)
    total_before   = sum(len(tc.original)   for tc in traced_chunks)
    total_after    = sum(len(tc.compressed) for tc in traced_chunks)
    ratio          = (1 - total_after / total_before) * 100 if total_before else 0

    print("\n┌" + "─" * W + "┐")
    print(_box_line("📦 PROMPT COMPRESSION LOG"))
    print("├" + "─" * W + "┤")

    if not any_compressed:
        print(_box_line(f"Context size: {total_before} chars  ≤  budget {CONTEXT_BUDGET} chars"))
        print(_box_line("✅ No compression needed — all chunks passed through unchanged."))
    else:
        print(_box_line(f"Context size: {total_before} chars  >  budget {CONTEXT_BUDGET} chars"))
        print(_box_line(f"Compressed to: {total_after} chars  ({ratio:.0f}% reduction)"))
        print("├" + "─" * W + "┤")
        for tc in traced_chunks:
            status = "compressed" if tc.was_compressed else "unchanged "
            before = len(tc.original)
            after  = len(tc.compressed)
            print(_box_line(f"Chunk {tc.chunk_id}: {before} → {after} chars  [{status}]"))
            if tc.was_compressed:
                # Show what was KEPT (compressed)
                kept_preview = tc.compressed[:120].replace("\n", " ").strip()
                print(_box_line(f'  ✂ kept : "{kept_preview}"'))
                # Show what was REMOVED (diff)
                removed_chars = before - after
                print(_box_line(f"  🗑 removed: ~{removed_chars} chars of irrelevant text"))
            print("├" + "─" * W + "┤")

    print("└" + "─" * W + "┘")


def log_sources(traced_chunks: list[TracedChunk]) -> None:
    """
    Print a full source traceability table.
    For each [Chunk N] the LLM may have cited, show:
      - the compressed text it actually saw
      - the complete original text from Supabase (ground truth)
    This lets you verify the answer against the raw source.
    """
    print("\n┌" + "─" * W + "┐")
    print(_box_line("📋 SOURCE TRACEABILITY"))
    print(_box_line("   (original Supabase text for each chunk the LLM saw)"))
    print("├" + "─" * W + "┤")

    for tc in traced_chunks:
        tag = "compressed" if tc.was_compressed else "full text "
        print(_box_line(f"[Chunk {tc.chunk_id}]  {len(tc.original)} chars  [{tag}]"))

        if tc.was_compressed:
            # Show compressed (what LLM saw) vs original (ground truth)
            print(_box_line("  LLM saw (compressed):"))
            for line in _wrap(tc.compressed, indent="    "):
                print(_box_line(line))
            print(_box_line("  ──────────────────────────────────────────────"))
            print(_box_line("  Original source (Supabase):"))
            # Show up to 300 chars of original to keep log readable
            original_preview = tc.original[:300].replace("\n", " ").strip()
            if len(tc.original) > 300:
                original_preview += f"  ... [{len(tc.original) - 300} more chars]"
            for line in _wrap(original_preview, indent="    "):
                print(_box_line(line))
        else:
            # No compression — LLM saw the full original
            print(_box_line("  LLM saw = original source (no compression applied):"))
            original_preview = tc.original[:300].replace("\n", " ").strip()
            if len(tc.original) > 300:
                original_preview += f"  ... [{len(tc.original) - 300} more chars]"
            for line in _wrap(original_preview, indent="    "):
                print(_box_line(line))

        print("├" + "─" * W + "┤")

    print("└" + "─" * W + "┘\n")


# ── 12. Chat loop ─────────────────────────────────────────────────────────────
def chat():
    print("=" * (W + 2))
    print("  DKU Bulletin Chatbot  (DSPy + Query Rewrite + Compression)")
    print("  Commands:  'quit' to exit | 'history' to inspect LM call")
    print("=" * (W + 2) + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if question.lower() == "history":
            dspy.inspect_history(n=1)
            continue

        result = rag(question=question)

        log_rewrite(
            original   = result.original_question,
            candidates = result.candidates,
            final      = result.final_rewrite,
        )
        log_compression(result.traced_chunks)

        print(f"Bot: {result.answer}\n")

        log_sources(result.traced_chunks)


if __name__ == "__main__":
    chat()