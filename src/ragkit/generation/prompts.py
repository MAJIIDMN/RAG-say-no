"""Prompt templates. Bump PROMPT_VERSION on any change: it is recorded in every eval run."""

from __future__ import annotations

from ragkit.core.schemas import ScoredChunk

PROMPT_VERSION = "v1"

ANSWER_GROUNDED = """\
You answer questions about software documentation using ONLY the numbered passages the user gives.

Rules:
- Every claim must be supported by the passages. For each claim, list the passage IDs (S1, S2, ...)
  that support it, each with a short quote copied character for character from that passage.
- Do not use outside knowledge, even when you are confident it is correct.
- Set status to "insufficient_context" and leave claims empty when the passages do not fully answer
  the question, when the question assumes something the passages contradict, or when you would have
  to guess. Say what is missing in missing_information. A partial or guessed answer is worse than
  no answer.
- Otherwise set status to "answered". Keep the answer to 1-4 claims, one sentence each.
"""

ANSWER_NAIVE = """\
You answer questions about software documentation using the numbered passages the user gives.

Always set status to "answered" and give your best answer as 1-4 claims, one sentence each. For each
claim, list the passage IDs (S1, S2, ...) it draws on, each with a short quote copied character for
character from that passage.
"""

SUFFICIENCY = """\
Decide whether the numbered passages contain enough information to fully and correctly answer the
question. Do not answer the question. Judge only from the passages, not from outside knowledge.
"""

VERIFY_CLAIM = """\
Decide whether the claim is fully supported by the evidence passages. It is supported only if every
part of it follows from the evidence alone; outside knowledge does not count.
"""


def format_context(context: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(context, start=1):
        c = sc.chunk
        header = c.section or c.title
        blocks.append(f"[S{i}] {header}\n{c.text}" if header else f"[S{i}]\n{c.text}")
    return "\n\n".join(blocks)


def answer_user(question: str, context: list[ScoredChunk]) -> str:
    return f"<passages>\n{format_context(context)}\n</passages>\n\n<question>{question}</question>"


def verify_user(claim: str, evidence: list[str]) -> str:
    ev = "\n\n".join(f"[E{i}]\n{t}" for i, t in enumerate(evidence, start=1))
    return f"<evidence>\n{ev}\n</evidence>\n\n<claim>{claim}</claim>"
