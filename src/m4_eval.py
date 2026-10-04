from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _enable_ragas_py314_compatibility() -> None:
    """Defer RAGAS task creation until an asyncio loop is running on Python 3.14.

    RAGAS 0.1.x calls ``asyncio.as_completed`` before ``asyncio.run``. Python
    3.14 rejects that pattern, unlike earlier Python releases. The adapter keeps
    RAGAS' bounded-concurrency behaviour but returns a lazy iterator, which is
    first consumed inside RAGAS' already-created event loop.
    """
    if sys.version_info < (3, 14):
        return

    import asyncio
    import ragas.executor as ragas_executor

    if getattr(ragas_executor, "_lab18_py314_compatible", False):
        return

    def lazy_as_completed(coros, max_workers):
        def iterator():
            if max_workers == -1:
                yield from asyncio.as_completed(coros)
                return

            semaphore = asyncio.Semaphore(max_workers)

            async def semaphore_coro(coro):
                async with semaphore:
                    return await coro

            yield from asyncio.as_completed([semaphore_coro(coro) for coro in coros])

        return iterator()

    ragas_executor.as_completed = lazy_as_completed
    ragas_executor._lab18_py314_compatible = True


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = (
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    )
    empty_result = {**{name: 0.0 for name in metric_names}, "per_question": []}

    if not questions:
        return empty_result
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        print("  ⚠️  RAGAS evaluation skipped: input lists must have the same length.")
        return empty_result
    # Unit tests must not send fixture text to a billable external LLM service.
    # The real pipeline is run outside pytest and evaluates normally.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return empty_result

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        _enable_ragas_py314_compatibility()

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        evaluation = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        dataframe = evaluation.to_pandas()

        def score(value) -> float:
            """Convert RAGAS/Pandas numeric values, treating missing values as zero."""
            try:
                value = float(value)
                return value if math.isfinite(value) else 0.0
            except (TypeError, ValueError):
                return 0.0

        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=score(row.get("faithfulness")),
                answer_relevancy=score(row.get("answer_relevancy")),
                context_precision=score(row.get("context_precision")),
                context_recall=score(row.get("context_recall")),
            )
            for _, row in dataframe.iterrows()
        ]

        aggregates = {
            name: (
                sum(getattr(item, name) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for name in metric_names
        }
        return {**aggregates, "per_question": per_question}
    except Exception as error:
        # RAGAS uses an LLM and embeddings under the hood. A missing API key,
        # temporary provider failure, or incompatible package should not stop the
        # retrieval pipeline from producing its report.
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return empty_result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metric_names = (
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    )

    failures = []
    for result in eval_results:
        scores = {metric: float(getattr(result, metric)) for metric in metric_names}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return sorted(failures, key=lambda failure: failure["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
