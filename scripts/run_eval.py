"""
Run evaluation.

Usage:
    python -m scripts.run_eval

For each case:
  1. Ingest the document into the in-memory store.
  2. Ask the question through the real agent.
  3. Score with Ragas: faithfulness + context_precision.
  4. For refusal cases, check the answer matches the canonical refusal.

Exits non-zero if any check fails. This is the shape you'd wire into CI.

Caveats (read these before trusting the numbers):
  - The grader uses the same LLM as the system under test. That's bad
    practice; the grader should be a stronger, independent model. Single
    key for the demo. Fix is a one-line change in _score().
  - 3 cases is not an eval set, it's a smoke test. A real eval set is 30+
    cases sourced from real user questions, refreshed quarterly.
  - Thresholds are picked from a single dry-run, not tuned. Re-tune after
    any prompt or model change.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402
from app.services.agent import REFUSAL, answer_question  # noqa: E402
from app.services.ingestion import ingest_document  # noqa: E402
from scripts.eval_cases import CASES, EvalCase  # noqa: E402


FAITHFULNESS_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7"))
CONTEXT_PRECISION_THRESHOLD = float(os.environ.get("CONTEXT_PRECISION_THRESHOLD", "0.5"))


def _run_case(case: EvalCase, store: DocumentStore) -> dict:
    record = ingest_document(text=case.document, store=store)
    result = answer_question(record=record, question=case.question)
    return {
        "name": case.name,
        "question": case.question,
        "answer": result["answer"],
        "contexts": [c["content"] for c in result["source_chunks"]],
        "ground_truth": case.ground_truth,
        "expect_refusal": case.expect_refusal,
        "short_circuited": result.get("short_circuited", False),
    }


def _score(runs: list[dict]) -> dict[str, dict[str, float]]:
    """
    Score with Ragas faithfulness + context_precision.

    Returns {case_name: {metric_name: score}}.

    Imports happen here so the script gives a clean error if ragas isn't
    installed, rather than blowing up at file load time.
    """
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import context_precision, faithfulness
    except ImportError as exc:
        print(
            "Ragas / datasets not installed. Install with: "
            "pip install ragas datasets",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    from app.services.agent import get_llm

    # NOTE: same model as the system under test. See module docstring.
    grader_llm = LangchainLLMWrapper(get_llm())
    grader_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Skip refusal cases — they're scored by exact-match below.
    # Also skip short-circuited runs since there are no contexts to score.
    scoreable = [r for r in runs if not r["expect_refusal"] and not r["short_circuited"]]
    if not scoreable:
        return {}

    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in scoreable
    ])
    result = evaluate(
        ds,
        metrics=[faithfulness, context_precision],
        llm=grader_llm,
        embeddings=grader_emb,
    )
    df = result.to_pandas()
    scores: dict[str, dict[str, float]] = {}
    for idx, r in enumerate(scoreable):
        scores[r["name"]] = {
            "faithfulness": float(df["faithfulness"].iloc[idx]),
            "context_precision": float(df["context_precision"].iloc[idx]),
        }
    return scores


def main() -> int:
    configure_logging("INFO")
    logger = get_logger("eval")
    store = DocumentStore()

    runs = []
    for case in CASES:
        logger.info("running_case", extra={"case_name": case.name})
        runs.append(_run_case(case, store))

    scores = _score(runs)

    print("\n" + "=" * 88)
    print(f"{'CASE':<26} {'METRIC':<22} {'SCORE':>10} {'THRESH':>10} {'VERDICT':>10}")
    print("=" * 88)

    failures = 0
    for run in runs:
        if run["expect_refusal"]:
            passed = run["answer"].strip() == REFUSAL
            verdict = "PASS" if passed else "FAIL"
            score_str = "exact-match" if passed else "mismatch"
            print(f"{run['name']:<26} {'refusal':<22} {score_str:>10} {'—':>10} {verdict:>10}")
            if not passed:
                failures += 1
        else:
            case_scores = scores.get(run["name"], {})
            for metric, threshold in [
                ("faithfulness", FAITHFULNESS_THRESHOLD),
                ("context_precision", CONTEXT_PRECISION_THRESHOLD),
            ]:
                s = case_scores.get(metric)
                s_str = f"{s:.3f}" if s is not None else "n/a"
                passed = s is not None and s >= threshold
                verdict = "PASS" if passed else "FAIL"
                print(f"{run['name']:<26} {metric:<22} {s_str:>10} {threshold:>10.2f} {verdict:>10}")
                if not passed:
                    failures += 1
    print("=" * 88)

    report_path = Path(__file__).parent / "eval_report.json"
    report_path.write_text(
        json.dumps({"runs": runs, "scores": scores}, indent=2, default=str)
    )
    print(f"Detailed report: {report_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
