"""Answer generation — RAG mode and LLM-only baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from . import client, format, prompts


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[int]
    citation_to_passage_id: dict[int, str]
    mode: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": self.citations,
            "citation_to_passage_id": self.citation_to_passage_id,
            "mode": self.mode,
        }


def generate_grounded(question: str, passages: Sequence[dict], model: str | None = None) -> GeneratedAnswer:
    messages = prompts.build_grounded_messages(question, passages)
    text = client.chat(messages, model=model)
    citations = format.find_citations(text)
    mapping = format.map_citations_to_ids(text, passages)
    return GeneratedAnswer(text=text, citations=citations, citation_to_passage_id=mapping, mode="rag")


def generate_llm_only(question: str, model: str | None = None) -> GeneratedAnswer:
    messages = prompts.build_llm_only_messages(question)
    text = client.chat(messages, model=model)
    return GeneratedAnswer(text=text, citations=[], citation_to_passage_id={}, mode="llm")


def stream_grounded_text(
    question: str,
    passages: Sequence[dict],
    model: str | None = None,
) -> Iterator[str]:
    messages = prompts.build_grounded_messages(question, passages)
    yield from client.chat_stream(messages, model=model)


def stream_llm_only_text(question: str, model: str | None = None) -> Iterator[str]:
    messages = prompts.build_llm_only_messages(question)
    yield from client.chat_stream(messages, model=model)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    from ..retrieval.search import retrieve

    parser = argparse.ArgumentParser(description="Quick CLI for source-grounded generation.")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--llm-only", action="store_true", help="Skip retrieval; baseline.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if not args.question:
        print("usage: python -m rag_turkish_law.generation.generate \"Soru?\"", file=sys.stderr)
        sys.exit(2)

    if args.llm_only:
        ans = generate_llm_only(args.question)
    else:
        hits = retrieve(args.question, k=args.k)
        ans = generate_grounded(args.question, [h.to_dict() for h in hits])
    print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
