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
    max_tokens  = 512,
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


class AnswerFromBulletin(dspy.Signature):
    """
    Answer student questions about the DKU Bulletin accurately and concisely.
    Base your answer only on the provided context.
    If the answer is not in the context, say so clearly.
    """
    context:  list[str] = dspy.InputField(desc="Relevant passages from the DKU Bulletin")
    question: str        = dspy.InputField(desc="The student's original question")
    answer:   str        = dspy.OutputField(desc="Accurate answer grounded in the context")


# ── 6. Query Rewriter Module ──────────────────────────────────────────────────
#
#   A standalone DSPy module just for rewriting.
#   Uses dspy.Predict (not ChainOfThought) because we want structured
#   list output, not open-ended reasoning prose.
#
class QueryRewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteQuery)

    def forward(self, raw_question: str) -> dspy.Prediction:
        return self.rewrite(raw_question=raw_question)


# ── 7. RAG Module with Query Rewriting ───────────────────────────────────────
#
#   Pipeline:
#     raw question
#       → QueryRewriter  (LLM call #1: fix typos, expand abbreviations)
#       → SupabaseRM     (embed final_rewrite, retrieve top-5 chunks)
#       → ChainOfThought (LLM call #2: generate grounded answer)
#
class DKUBulletinRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewriter       = QueryRewriter()
        self.generate_answer = dspy.ChainOfThought(AnswerFromBulletin)

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1: rewrite the raw question
        rewrite_result = self.rewriter(raw_question=question)

        candidates    = rewrite_result.candidate_rewrites   # list[str]
        final_rewrite = rewrite_result.final_rewrite        # str

        # Step 2: retrieve using the best rewrite
        context = retriever(final_rewrite)

        # Step 3: generate answer using the original question (not the rewrite)
        #         so the bot's tone stays natural for the user
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            original_question = question,
            candidates        = candidates,
            final_rewrite     = final_rewrite,
            context           = context,
            answer            = prediction.answer,
        )


# ── 8. Instantiate ────────────────────────────────────────────────────────────
rag = DKUBulletinRAG()


# ── 9. Query rewrite logger ───────────────────────────────────────────────────
def log_rewrite(original: str, candidates: list[str], final: str) -> None:
    """Print a structured log of the rewrite step to the terminal."""
    print("\n" + "┌" + "─" * 58 + "┐")
    print(f"│  🔍 QUERY REWRITE LOG{' ' * 36}│")
    print("├" + "─" * 58 + "┤")
    print(f"│  Original : {original[:50]:<50}│" if len(original) <= 50
          else f"│  Original : {original[:47]}...│")
    print("├" + "─" * 58 + "┤")
    print("│  Candidates:                                          │")
    for i, c in enumerate(candidates, 1):
        label = f"  {i}. "
        line  = (label + c)[:56]
        print(f"│  {line:<56}│")
    print("├" + "─" * 58 + "┤")
    used = f"  ✅ Used    : {final}"
    print(f"│{used[:58]:<58}│")
    print("└" + "─" * 58 + "┘\n")


# ── 10. Chat loop ─────────────────────────────────────────────────────────────
def chat():
    print("=" * 60)
    print("  DKU Bulletin Chatbot  (DSPy + Query Rewrite + Supabase)")
    print("  Commands:  'quit' to exit | 'history' to inspect LM call")
    print("=" * 60 + "\n")

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

        # Log the rewrite step before printing the answer
        log_rewrite(
            original   = result.original_question,
            candidates = result.candidates,
            final      = result.final_rewrite,
        )

        print(f"Bot: {result.answer}\n")


if __name__ == "__main__":
    chat()