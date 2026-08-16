"""
Phase 10 — Prompt Engineering.

Problem:
    An LLM given only a bare question will answer from its own pretrained
    knowledge, which may be outdated, generic, or simply wrong for the
    specific library version in the documentation. We need to force the
    model to ground its answer in the retrieved chunks and nothing else.

Why this specific prompt structure:
    The prompt is split into three explicit sections — instructions,
    context, question — so the model can't confuse retrieved documentation
    with the question itself. The instruction block explicitly tells the
    model to (a) only use the provided context and (b) say so plainly if the
    answer isn't in the context, rather than guessing. This second
    instruction is what makes the retrieval guardrail in retrieval.py and
    the prompt work together: even if a borderline-relevant chunk slips
    through, the model still has a documented "don't know" escape hatch.

Why not just concatenate context + question with no instructions:
    Instruction-tuned models (Mistral-7B-Instruct, Llama-3-Instruct) are
    specifically trained to follow explicit natural-language instructions
    in the prompt. Skipping the instruction block and hoping the model
    infers "answer only from this text" from context alone is unreliable —
    it will often blend in outside knowledge, especially for well-known
    libraries like pandas or NumPy where the model has strong priors.
"""

from langchain_core.documents import Document

SYSTEM_INSTRUCTIONS = """You are a technical documentation assistant. Answer the user's \
question using ONLY the documentation excerpts provided in the context below.

Rules:
1. Do not use any knowledge outside the given context, even if you know the answer.
2. If the context does not contain enough information to answer, respond exactly with: \
"I could not find this in the provided documentation."
3. When you use a piece of context, keep code examples verbatim if the question is about code.
4. Be concise and technically precise."""


def build_prompt(question: str, context_chunks: list[Document]) -> str:
    """Combine retrieved context + question into a single grounded prompt."""
    context_blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        context_blocks.append(
            f"[Excerpt {i} | source: {source}, page: {page}]\n{chunk.page_content}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_INSTRUCTIONS}

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Question: {question}

Answer:"""
    return prompt


if __name__ == "__main__":
    dummy_chunks = [
        Document(
            page_content="df.merge(other_df, on='key') combines two DataFrames on a shared column.",
            metadata={"source": "pandas_docs.pdf", "page": 12},
        )
    ]
    print(build_prompt("How do I merge two DataFrames?", dummy_chunks))
