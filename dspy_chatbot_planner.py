"""
dspy_chatbot.py
===============
DKU Bulletin chatbot with a full DSPy pipeline:

  Step 0  Complexity Check    — is this a simple or complex question?
  Step 1  Query Rewrite       — fix typos / expand abbreviations
  Step 2  Planner             — for complex queries, decompose into sub-questions
  Step 3  Iterative Retrieval — retrieve from Supabase for each sub-question
  Step 4  Prompt Compression  — trim context if it exceeds the token budget
  Step 5  Final Answer        — ChainOfThought with [Chunk N] citations

Simple questions (e.g. "what is the GPA requirement?") skip the planner
and use a single retrieval pass.

Complex questions (e.g. "compare CS and Applied Maths — which has more
career flexibility?") are decomposed into sub-questions, each retrieved
independently, then synthesised into one final answer.

Install:
  pip install dspy llama-index llama-index-vector-stores-supabase
               llama-index-embeddings-huggingface vecs python-dotenv
"""

import os
import sys
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

import dspy
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy.exc import OperationalError


# ── 1. Credentials ────────────────────────────────────────────────────────────
load_dotenv()
DB_URL   = os.getenv("SUPABASE_DB_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
if not DB_URL:   print("ERROR: SUPABASE_DB_URL not found"); sys.exit(1)
if not HF_TOKEN: print("ERROR: HF_TOKEN not found");        sys.exit(1)


# ── 2. DSPy LM ────────────────────────────────────────────────────────────────
lm = dspy.LM(
    model="openai/meta-llama/Llama-3.1-8B-Instruct:novita",
    api_base="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    max_tokens=2048,
    temperature=0.2,
)
dspy.configure(lm=lm)
print("✅  dspy.LM configured → Llama-3.1-8B via HuggingFace router")


# ── 3. Embeddings + Supabase ──────────────────────────────────────────────────
print("Loading embedding model and connecting to Supabase...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
print("✅  Ready.\n")


# ── 4. Retriever with auto-reconnect ──────────────────────────────────────────
def _build_retriever(db_url: str, k: int):
    vs = SupabaseVectorStore(
        postgres_connection_string=db_url,
        collection_name="dku_bulletin",
        dimension=384,
    )
    return VectorStoreIndex.from_vector_store(vs).as_retriever(similarity_top_k=k)


class SupabaseRM:
    def __init__(self, db_url: str, k: int = 5):
        self.db_url = db_url
        self.k = k
        self._retriever = _build_retriever(db_url, k)

    def _reconnect(self):
        print("\n[Supabase] Reconnecting...", flush=True)
        time.sleep(1)
        self._retriever = _build_retriever(self.db_url, self.k)
        print("[Supabase] Reconnected.\n", flush=True)

    def __call__(self, query: str, k: int = None) -> list[str]:
        self._retriever.similarity_top_k = k or self.k
        try:
            nodes = self._retriever.retrieve(query)
        except Exception as e:
            if "server closed the connection" in str(e) or "OperationalError" in type(e).__name__:
                self._reconnect()
                nodes = self._retriever.retrieve(query)
            else:
                raise
        return [node.text for node in nodes]


retriever = SupabaseRM(db_url=DB_URL, k=5)


# ── 5. Data classes ───────────────────────────────────────────────────────────
@dataclass
class TracedChunk:
    """One retrieved chunk with full before/after traceability.
    chunk_id matches the [Chunk N] label the answer LLM sees and cites."""
    chunk_id:       int
    sub_question:   str   # which sub-question triggered this retrieval
    original:       str   # verbatim text from Supabase (ground truth)
    compressed:     str   # what the answer LLM actually received
    was_compressed: bool


@dataclass
class SubQuestionResult:
    """Retrieval results for one planned sub-question."""
    sub_question: str
    chunks:       list[TracedChunk] = field(default_factory=list)


# ── 6. Signatures ─────────────────────────────────────────────────────────────

class ClassifyComplexity(dspy.Signature):
    """
    Decide whether a student's question requires SIMPLE or COMPLEX retrieval.

    SIMPLE  : answered from a single section of the bulletin (one major,
              one policy, one requirement list).
              Examples:
                "What are the CS major requirements?"
                "What is the GPA threshold for academic probation?"

    COMPLEX : requires comparing multiple majors/policies, tracing
              prerequisites across departments, or synthesising information
              from several unrelated sections.
              Examples:
                "Compare CS and Applied Maths — which has more flexibility?"
                "If I double-major in Biology and Chemistry, what extra
                 courses do I need beyond each major's own requirements?"

    Return exactly one word: SIMPLE or COMPLEX.
    """
    question:   str = dspy.InputField(desc="The student's question")
    complexity: str = dspy.OutputField(desc="Exactly one word: SIMPLE or COMPLEX")


class RewriteQuery(dspy.Signature):
    """
    Optimize a student's question for vector search in a university bulletin.
    Produce 3 distinct rewrites and select the best one.

    Rules:
    - Expand abbreviations ("CS" → "Computer Science", "appl maths" → "Applied Mathematics")
    - Fix typos
    - Each candidate approaches the topic from a slightly different angle
    - Keep queries factual and concise
    """
    raw_question:       str       = dspy.InputField(desc="Student's original unedited question")
    candidate_rewrites: list[str] = dspy.OutputField(desc="Exactly 3 rewritten query candidates")
    final_rewrite:      str       = dspy.OutputField(desc="The single best rewrite for retrieval")


class DecomposePlan(dspy.Signature):
    """
    Break a complex university bulletin question into 2–4 focused sub-questions.
    Each sub-question must be independently searchable in the bulletin database.
    Together they must cover everything needed to fully answer the original question.

    Rules:
    - Each sub-question targets ONE specific piece of information
    - Use precise academic terms (full major names, exact policy names)
    - Order sub-questions so facts are gathered before comparisons are made
    - Do NOT include a synthesis sub-question — synthesis happens automatically
      after all retrieval is done
    """
    question:      str       = dspy.InputField(desc="The complex student question to decompose")
    sub_questions: list[str] = dspy.OutputField(desc="2–4 focused, independently searchable sub-questions")


class CompressChunk(dspy.Signature):
    """
    Extract ONLY the sentences from a passage that directly answer the question.
    - Keep original wording — do not paraphrase
    - Preserve numbers, course codes, credit counts, policy names exactly
    - Return an empty string if nothing is relevant
    - Output must be shorter than the input
    """
    question:           str = dspy.InputField(desc="The student's question or sub-question")
    passage:            str = dspy.InputField(desc="One retrieved passage from the DKU Bulletin")
    compressed_passage: str = dspy.OutputField(desc="Only the relevant sentences, verbatim")


class AnswerFromBulletin(dspy.Signature):
    """
    Answer the student's question using ONLY the provided context passages.
    If the answer cannot be found in the context, say so clearly.

    IMPORTANT: Every factual claim in your answer MUST be followed by a
    [Chunk N] citation matching the label on the passage it came from.
    Example: "CS requires 128 credits [Chunk 1] including COMPSCI 101 [Chunk 3]."
    """
    context:  list[str] = dspy.InputField(desc="Passages from the DKU Bulletin, each labelled [Chunk N]")
    question: str        = dspy.InputField(desc="The student's original question")
    answer:   str        = dspy.OutputField(desc="Answer with inline [Chunk N] citations for every factual claim")


# ── 7. DSPy Modules ───────────────────────────────────────────────────────────

class QueryRewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteQuery)

    def forward(self, raw_question: str) -> dspy.Prediction:
        return self.rewrite(raw_question=raw_question)


CONTEXT_BUDGET = 3000  # chars; ~750 tokens — safe for Llama-3.1-8B 8k window

class PromptCompressor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.compress = dspy.Predict(CompressChunk)

    def forward(self, question: str, chunks: list[str],
                sub_question: str = "", id_offset: int = 0) -> list[TracedChunk]:
        total_chars = sum(len(c) for c in chunks)
        label = sub_question or question
        traced = []

        if total_chars <= CONTEXT_BUDGET:
            for i, chunk in enumerate(chunks):
                traced.append(TracedChunk(
                    chunk_id=id_offset + i + 1, sub_question=label,
                    original=chunk, compressed=chunk, was_compressed=False,
                ))
            return traced

        for i, chunk in enumerate(chunks):
            result     = self.compress(question=label, passage=chunk)
            compressed = result.compressed_passage.strip()
            if not compressed or len(compressed) >= len(chunk):
                compressed, was_compressed = chunk, False
            else:
                was_compressed = True
            traced.append(TracedChunk(
                chunk_id=id_offset + i + 1, sub_question=label,
                original=chunk, compressed=compressed,
                was_compressed=was_compressed,
            ))
        return traced


# ── 8. Main RAG Module ────────────────────────────────────────────────────────
#
#   classify(question)
#     │
#     ├─ SIMPLE ──► rewrite → retrieve(×1) → compress → answer
#     │
#     └─ COMPLEX ─► rewrite → planner decomposes into sub-questions
#                              for each sub-question:
#                                retrieve(×1) → compress
#                              merge all TracedChunks → answer
#
class DKUBulletinRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify        = dspy.Predict(ClassifyComplexity)
        self.rewriter        = QueryRewriter()
        self.planner         = dspy.Predict(DecomposePlan)
        self.compressor      = PromptCompressor()
        self.generate_answer = dspy.ChainOfThought(AnswerFromBulletin)

    def _simple_path(self, question, final_rewrite):
        raw_chunks    = retriever(final_rewrite)
        traced_chunks = self.compressor(
            question=question, chunks=raw_chunks, sub_question=final_rewrite,
        )
        return [], traced_chunks  # no sub_results for simple

    def _complex_path(self, question, final_rewrite):
        plan          = self.planner(question=final_rewrite)
        sub_questions = plan.sub_questions

        all_traced: list[TracedChunk]       = []
        sub_results: list[SubQuestionResult] = []
        chunk_counter = 0

        for sq in sub_questions:
            raw_chunks = retriever(sq)
            traced     = self.compressor(
                question=question, chunks=raw_chunks,
                sub_question=sq, id_offset=chunk_counter,
            )
            chunk_counter += len(traced)
            all_traced.extend(traced)
            sub_results.append(SubQuestionResult(sub_question=sq, chunks=traced))

        return sub_results, all_traced

    def forward(self, question: str) -> dspy.Prediction:
        # Step 0: classify
        complexity = self.classify(question=question).complexity.strip().upper()
        is_complex = "COMPLEX" in complexity

        # Step 1: rewrite
        rw            = self.rewriter(raw_question=question)
        candidates    = rw.candidate_rewrites
        final_rewrite = rw.final_rewrite

        # Step 2–3: route
        if is_complex:
            sub_results, traced_chunks = self._complex_path(question, final_rewrite)
        else:
            sub_results, traced_chunks = self._simple_path(question, final_rewrite)

        # Step 4: build labelled context
        context_for_llm = [
            f"[Chunk {tc.chunk_id}] {tc.compressed}"
            for tc in traced_chunks
            if tc.compressed.strip()
        ]

        # Step 5: generate answer
        prediction = self.generate_answer(
            context=context_for_llm,
            question=question,
        )

        return dspy.Prediction(
            original_question = question,
            is_complex        = is_complex,
            candidates        = candidates,
            final_rewrite     = final_rewrite,
            sub_results       = sub_results,
            traced_chunks     = traced_chunks,
            answer            = prediction.answer,
        )


# ── 9. Instantiate ────────────────────────────────────────────────────────────
rag = DKUBulletinRAG()


# ── 10. Loggers ───────────────────────────────────────────────────────────────
W = 64

def _box_line(text: str) -> str:
    return f"│  {text:<{W - 4}}│"

def _wrap(text: str, indent: str = "    ") -> list[str]:
    width = W - 6
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    return [indent + l for l in lines] if lines else [indent + text[:width]]


def log_rewrite(original: str, candidates: list[str], final: str) -> None:
    print("\n┌" + "─" * W + "┐")
    print(_box_line("🔍 QUERY REWRITE LOG"))
    print("├" + "─" * W + "┤")
    disp = original if len(original) <= W - 16 else original[:W - 19] + "..."
    print(_box_line(f"Original : {disp}"))
    print("├" + "─" * W + "┤")
    print(_box_line("Candidates:"))
    for i, c in enumerate(candidates, 1):
        d = c if len(c) <= W - 10 else c[:W - 13] + "..."
        print(_box_line(f"  {i}. {d}"))
    print("├" + "─" * W + "┤")
    fd = final if len(final) <= W - 14 else final[:W - 17] + "..."
    print(_box_line(f"✅ Used   : {fd}"))
    print("└" + "─" * W + "┘")


def log_plan(is_complex: bool, sub_results: list[SubQuestionResult]) -> None:
    print("\n┌" + "─" * W + "┐")
    if not is_complex:
        print(_box_line("🗺  PLANNER LOG"))
        print("├" + "─" * W + "┤")
        print(_box_line("Classified: SIMPLE — planner skipped."))
        print(_box_line("Single retrieval pass used."))
        print("└" + "─" * W + "┘")
        return

    print(_box_line(f"🗺  PLANNER LOG  ── COMPLEX query detected"))
    print("├" + "─" * W + "┤")
    print(_box_line(f"Decomposed into {len(sub_results)} sub-questions:"))
    print("├" + "─" * W + "┤")
    for i, sr in enumerate(sub_results, 1):
        sq_d = sr.sub_question if len(sr.sub_question) <= W - 14 \
               else sr.sub_question[:W - 17] + "..."
        chunk_ids = ", ".join(f"[Chunk {tc.chunk_id}]" for tc in sr.chunks)
        print(_box_line(f"  Sub-Q {i}: {sq_d}"))
        print(_box_line(f"           → {chunk_ids}"))
    print("└" + "─" * W + "┘")


def log_compression(traced_chunks: list[TracedChunk]) -> None:
    any_comp     = any(tc.was_compressed for tc in traced_chunks)
    total_before = sum(len(tc.original)   for tc in traced_chunks)
    total_after  = sum(len(tc.compressed) for tc in traced_chunks)
    ratio        = (1 - total_after / total_before) * 100 if total_before else 0

    print("\n┌" + "─" * W + "┐")
    print(_box_line("📦 PROMPT COMPRESSION LOG"))
    print("├" + "─" * W + "┤")

    if not any_comp:
        print(_box_line(f"Total context: {total_before} chars  ≤  budget {CONTEXT_BUDGET} chars"))
        print(_box_line("✅ No compression needed — all chunks unchanged."))
    else:
        print(_box_line(f"Total context: {total_before} chars  >  budget {CONTEXT_BUDGET} chars"))
        print(_box_line(f"Compressed to: {total_after} chars  ({ratio:.0f}% reduction)"))
        print("├" + "─" * W + "┤")
        for tc in traced_chunks:
            status = "compressed" if tc.was_compressed else "unchanged "
            print(_box_line(
                f"  Chunk {tc.chunk_id:>2}: {len(tc.original)} → {len(tc.compressed)} chars  [{status}]"
            ))
            if tc.was_compressed:
                kept = tc.compressed[:90].replace("\n", " ").strip()
                print(_box_line(f'    ✂ kept: "{kept}..."'))
    print("└" + "─" * W + "┘")


def log_sources(traced_chunks: list[TracedChunk]) -> None:
    print("\n┌" + "─" * W + "┐")
    print(_box_line("📋 SOURCE TRACEABILITY"))
    print(_box_line("   Verify every [Chunk N] citation against its Supabase source"))
    print("├" + "─" * W + "┤")

    for tc in traced_chunks:
        tag  = "compressed" if tc.was_compressed else "full text"
        sq_d = tc.sub_question if len(tc.sub_question) <= W - 22 \
               else tc.sub_question[:W - 25] + "..."
        print(_box_line(f"[Chunk {tc.chunk_id}]  {len(tc.original)} chars  [{tag}]"))
        print(_box_line(f"  from: \"{sq_d}\""))

        if tc.was_compressed:
            print(_box_line("  LLM saw (compressed):"))
            for ln in _wrap(tc.compressed[:200].replace("\n", " ")):
                print(_box_line(ln))
            print(_box_line("  ── original Supabase text: ─────────────────────────"))
        else:
            print(_box_line("  LLM saw = full original text:"))

        orig = tc.original[:280].replace("\n", " ").strip()
        if len(tc.original) > 280:
            orig += f"  [...{len(tc.original)-280} more chars]"
        for ln in _wrap(orig):
            print(_box_line(ln))

        print("├" + "─" * W + "┤")
    print("└" + "─" * W + "┘\n")


# ── 11. Chat loop ─────────────────────────────────────────────────────────────
def chat():
    print("=" * (W + 2))
    print("  DKU Bulletin Chatbot")
    print("  (DSPy Planner + Query Rewrite + Compression + Traceability)")
    print("  Commands: 'quit' to exit  |  'history' to inspect last LM call")
    print("=" * (W + 2))
    print()
    print("💡 Example complex query:")
    print('   "Compare the Computer Science and Applied Mathematics majors.')
    print('    Which one gives more flexibility for someone interested in')
    print('    both programming and statistics?"\n')

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
        log_plan(result.is_complex, result.sub_results)
        log_compression(result.traced_chunks)

        print(f"\nBot: {result.answer}\n")

        log_sources(result.traced_chunks)


if __name__ == "__main__":
    chat()