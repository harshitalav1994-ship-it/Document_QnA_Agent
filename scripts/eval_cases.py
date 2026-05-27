"""
Eval test cases.

Three deliberately different cases:
  1. Direct factual answer  -> tests faithful retrieval + answering.
  2. Multi-fact synthesis   -> tests stitching across chunks.
  3. Unanswerable question  -> tests graceful refusal.

In a real system this set should be 30+ cases sourced from real user
questions and graded by both automated metrics and humans.
"""
from dataclasses import dataclass


@dataclass
class EvalCase:
    name: str
    document: str
    question: str
    # `ground_truth` describes what a good answer looks like; Ragas uses it
    # for answer-correctness style metrics. Faithfulness itself only needs
    # the answer and the retrieved contexts.
    ground_truth: str
    expect_refusal: bool = False


SOLAR_DOC = """\
The Sun is the star at the centre of the Solar System. It has a diameter of
approximately 1.39 million kilometres, about 109 times that of Earth. Its mass
is about 330,000 times that of Earth and accounts for roughly 99.86% of the
total mass of the Solar System. The Sun is composed primarily of hydrogen
(about 73%) and helium (about 25%). It formed approximately 4.6 billion years
ago from the gravitational collapse of matter within a region of a large
molecular cloud. Most of this matter gathered in the centre, whereas the rest
flattened into an orbiting disk that became the Solar System.
"""

CASES: list[EvalCase] = [
    EvalCase(
        name="direct_fact",
        document=SOLAR_DOC,
        question="What percentage of the Solar System's mass is the Sun?",
        ground_truth="The Sun accounts for roughly 99.86% of the total mass of the Solar System.",
    ),
    EvalCase(
        name="multi_fact_synthesis",
        document=SOLAR_DOC,
        question="What is the Sun mostly made of and how old is it?",
        ground_truth=(
            "The Sun is composed primarily of hydrogen (~73%) and helium (~25%), "
            "and it formed approximately 4.6 billion years ago."
        ),
    ),
    EvalCase(
        name="unanswerable",
        document=SOLAR_DOC,
        question="What is the surface temperature of the Sun in Celsius?",
        ground_truth="I cannot answer this question from the provided document.",
        expect_refusal=True,
    ),
]
