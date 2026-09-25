"""Checking an answer against its evidence, deterministically, before anyone reads it.

Two rules a small model breaks and code can check: every figure in the answer must be
one the tools produced (or the question stated), and every citation must name evidence
the tools returned - the query result, the prediction or a passage. A judge model
would be a second opinion from the same kind of model; these are checks. What they
cannot see - a claim a passage does not support - is left to the evaluation, where a
judge is admitted only after its agreement with people is measured.
"""

import re
from collections.abc import Collection

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)")


def figures(text: str) -> list[tuple[str, float, int, bool]]:
    """Every number in `text`: as written, its value, its decimals, and whether it is a
    percentage. A citation such as [c2] is not a number."""
    return [
        (match[0], float(match[1].replace(",", "") + (f".{match[2]}" if match[2] else "")),
         len(match[2] or ""), bool(match[3]))
        for match in _NUMBER.finditer(text)
    ]  # fmt: skip


def cited_ids(answer: str, citations: Collection[str]) -> set[str]:
    """Every id the answer cites, in its text or its list - "[c2]" and "c2" alike: a model
    that brackets the ids in the list means the same thing."""
    listed = {citation.strip().strip("[]").strip() for citation in citations}
    return {c for c in listed if c} | set(re.findall(r"\[(\w+)\]", answer))


def problems(
    answer: str, citations: Collection[str], evidence: str, sources: Collection[str]
) -> list[str]:
    """What is wrong with an answer, in words the model can act on; empty when nothing.
    `sources` are the ids of the evidence it was given: "sql", "prediction", "c1"..."""
    found: list[str] = []
    cited = cited_ids(answer, citations)
    unknown = sorted(cited - set(sources))
    if unknown:
        found.append(f"Cited {', '.join(unknown)}, which is not evidence you were given.")
    if sources and not cited:
        found.append("Cite the evidence each statement comes from, by its id in brackets.")
    known = [value for _, value, _, _ in figures(evidence)]
    for written, value, decimals, percent in figures(answer):
        candidates = known + [k * 100 for k in known] if percent else known
        if not any(round(k, decimals) == value for k in candidates):
            found.append(f"The figure {written} is not in the evidence; use only figures it gives.")
    return found
