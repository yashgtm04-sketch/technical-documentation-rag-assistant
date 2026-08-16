"""
Phase 12 — Source Attribution, and overall pipeline orchestration.

Problem:
    Every phase so far (loading, chunking, embedding, retrieval, prompting,
    generation) is implemented as an independent, testable module. Something
    needs to wire them together into the single call the Streamlit app (and
    the eval script) actually needs: "given a question, return an answer
    plus its sources."

Why a thin orchestrator class instead of putting this logic in app.py:
    Keeping this logic out of the Streamlit file means it can be unit
    tested and reused by eval/run_eval.py without importing Streamlit at
    all. It also means the UI layer only ever deals with a single
    `AnswerResult` object instead of poking at retrieval internals.

Source attribution's role in reducing hallucination:
    Showing the user exactly which document, page, and passage the answer
    came from turns the system from a black box into something verifiable.
    A user (or interviewer) can check the cited passage against the answer
    in seconds — this transparency is what "source-backed" means in
    practice, not just a UI nicety.
"""

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from src.prompt_builder import build_prompt
from src.retrieval import retrieve, RetrievalResult
from src.llm import generate_answer, LLMGenerationError

NOT_FOUND_MESSAGE = "I could not find this in the provided documentation."


@dataclass
class SourceChunk:
    source: str
    page: int
    score: float
    text: str


@dataclass
class AnswerResult:
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    grounded: bool = True  # False when the guardrail blocked generation


def answer_question(question: str, vector_store: FAISS) -> AnswerResult:
    """Run the full retrieval -> guardrail -> prompt -> generation pipeline for one question."""
    retrieval_result: RetrievalResult = retrieve(question, vector_store)

    sources = [
        SourceChunk(
            source=chunk.metadata.get("source", "unknown"),
            page=chunk.metadata.get("page", -1),
            score=score,
            text=chunk.page_content,
        )
        for chunk, score in zip(retrieval_result.chunks, retrieval_result.scores)
    ]

    # Phase 9 guardrail: if the best match isn't confident, don't even call the LLM.
    if not retrieval_result.is_confident:
        return AnswerResult(answer=NOT_FOUND_MESSAGE, sources=sources, grounded=False)

    prompt = build_prompt(question, retrieval_result.chunks)

    try:
        answer_text = generate_answer(prompt)
    except LLMGenerationError as exc:
        return AnswerResult(answer=f"Error generating answer: {exc}", sources=sources, grounded=False)

    return AnswerResult(answer=answer_text, sources=sources, grounded=True)
