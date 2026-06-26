"""
LLM-as-a-Judge evaluation for retrieval system quality.

Evaluates quality by asking an LLM to judge how useful a retrieved span is for
answering the query, compared to the annotated gold span, as well as if the 
retrieved span is factually consistent.

Grading scale:
0 = Factually incorrect OR contradicts the gold span(s), even if relevant
1 = Factually consistent but mostly irrelevant or contains no useful information for the query
2 = Factually consistent and partially useful, but incomplete or missing key 
information from the gold span(s)
3 = Factually consistent and highly useful; captures the key facts from the 
gold span(s) needed to answer the query

Usage:
    python3 llm_judge_eval.py --input <PATH> \
        --gold-spans <PATH> \
        --model <MODEL> \
        --api-key <KEY> \
        --rank <INT> \
        --output-dir <PATH>
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from mistralai import Mistral
from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = "You are an expert assessor for biomedical information retrieval."

JUDGE_PROMPT = (
"""Given a query, one or more gold-standard reference spans, and a retrieved span, evaluate the retrieved span based on BOTH:

1. Factual consistency with the gold span(s) (does it contain correct information?)
2. Usefulness for answering the query

When multiple gold spans are provided, treat them collectively as the reference: the retrieved span should be judged against the union of key facts expressed across all gold spans.

Scoring rules (STRICT):

* 0 = Factually incorrect OR contradicts the gold span(s), even if relevant
* 1 = Factually consistent but mostly irrelevant or contains no useful information for the query
* 2 = Factually consistent and partially useful, but incomplete or missing key information from the gold span(s)
* 3 = Factually consistent and highly useful; captures the key facts from the gold span(s) needed to answer the query

Important guidelines:

* Factual correctness is more important than relevance
* Any contradiction or clearly wrong statement → score 0
* Do NOT reward spans that are only topically similar but contain no concrete facts
* Prefer spans that contain specific, correct, and relevant information

Reply with ONLY a single digit: 0, 1, 2, or 3. No explanation.
"""
)


def build_user_message(query: str, gold_spans: list[str], retrieved_span: str) -> str:
    """Format the user message sent to the LLM judge.

    Args:
        query: The retrieval query.
        gold_spans: The annotated gold-standard text spans (one or more).
        retrieved_span: The text span retrieved by the system.

    Returns:
        Formatted prompt string containing judge instructions, query, gold
        span(s), and retrieved span.
    """
    if len(gold_spans) == 1:
        gold_section = f"## Gold span: {gold_spans[0]}"
    else:
        numbered = "\n\n".join(
            f"### Gold span {i}: {span}" for i, span in enumerate(gold_spans, 1)
        )
        gold_section = f"## Gold spans ({len(gold_spans)} total):\n\n{numbered}"

    return (
        f"{JUDGE_PROMPT}\n\n"
        f"## Query: {query}\n\n"
        f"{gold_section}\n\n"
        f"## Retrieved span: {retrieved_span}"
    )


def call_openai(api_key: str, model: str, system: str, user_msg: str,
                temperature: float, top_p: float) -> str:
    """Send a request to the OpenAI API using the official SDK.

    Args:
        api_key: OpenAI API key.
        model: Model identifier (e.g. "gpt-5").
        system: System prompt content.
        user_msg: User message content.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        The assistant's response text, stripped of whitespace.
    """
    client = OpenAI(api_key=api_key)
    return client.responses.create(
        model=model,
        instructions=system,
        input=user_msg,
        temperature=temperature,
        top_p=top_p,
    ).output_text.strip()


def call_mistral(api_key: str, model: str, system: str, user_msg: str,
                 temperature: float, top_p: float) -> str:
    """Send a chat completion request to the Mistral AI API using the official SDK.

    Args:
        api_key: Mistral API key.
        model: Model identifier (e.g. "mistral-large-latest").
        system: System prompt content.
        user_msg: User message content.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        The assistant's response text, stripped of whitespace.
    """
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content.strip()


def call_qwen(api_key: str, model: str, system: str, user_msg: str,
              temperature: float, top_p: float) -> str:
    """Send a chat completion request to Qwen via Alibaba Cloud DashScope.

    Args:
        api_key: DashScope API key.
        model: Model identifier (e.g. "qwen3-max").
        system: System prompt content.
        user_msg: User message content.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        The assistant's response text, stripped of whitespace.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    return completion.choices[0].message.content.strip()


# model name -> (caller function, default API model string)
MODEL_REGISTRY = {
    "gpt5.4": (call_openai, "gpt-5.4-2026-03-05"),
    "mistral-medium": (call_mistral, "mistral-medium-2508"),
    "mistral-large": (call_mistral, "mistral-large-2512"),
    "qwen3-max": (call_qwen, "qwen3-max-2026-01-23"),
}


def parse_score(raw: str) -> int | None:
    """Extract the first digit 0-3 from the LLM response.

    Args:
        raw: Raw text response from the LLM.

    Returns:
        Integer score (0-3) if found, None otherwise.
    """
    m = re.search(r"[0-3]", raw)
    return int(m.group()) if m else None


def run_evaluation(args: argparse.Namespace):
    """Run the full LLM-as-a-Judge evaluation pipeline.

    Iterates over all entries in the retrieval results, queries the selected
    LLM judge for each entry, and writes judgments continuously to disk.
    Produces a summary file with mean score and score distribution at the end.
    Supports resumption from a partially completed judgments file.

    Args:
        args: Parsed CLI arguments containing input path, model, api_key,
              temperature, top_p, rank, and output_dir.
    """
    model_key = args.model.lower()

    if model_key not in MODEL_REGISTRY:
        sys.exit(
            f"Unknown model '{args.model}'. Choose from: {', '.join(MODEL_REGISTRY.keys())}"
        )
    caller_fn, api_model = MODEL_REGISTRY[model_key]

    # Load retrieval results
    with open(args.input, "r") as f:
        data = json.load(f)

    # Load gold spans from the test set, keyed by PMCID
    gold_spans_by_pmcid: dict[str, list[str]] = {}
    with open(args.gold_spans, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gold_spans_by_pmcid[record["PMCID"]] = record["positives"]

    eval_rank = args.rank

    # Prepare output paths
    input_stem = Path(args.input).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / f"{input_stem}_rank{eval_rank}_{model_key}_judgments.json"
    summary_path = out_dir / f"{input_stem}_rank{eval_rank}_{model_key}_summary.json"

    # Load existing judgments to allow resumption
    existing_judgments = []
    already_judged: set[tuple[str, str]] = set()

    if judgments_path.exists():
        with open(judgments_path, "r") as f:
            existing_judgments = json.load(f)
        already_judged = {(j["PMCID"], j["query"]) for j in existing_judgments}
        print(f"Resuming: {len(already_judged)} entries already judged.")

    judgments = list(existing_judgments)
    scores: list[int] = [j["llm_score"] for j in existing_judgments if j["llm_score"] is not None]
    skipped_no_gold = 0
    skipped_no_rank = 0
    errors = 0

    pbar = tqdm(data, desc="Judging", unit="entry")

    for entry in pbar:
        pmcid = entry["PMCID"]
        query = entry["query"]
        if (pmcid, query) in already_judged:
            continue

        rankings = entry["top_50_ranking"]

        # Find the span at the requested evaluation rank
        eval_span_item = next((r for r in rankings if r["rank"] == eval_rank), None)

        if eval_span_item is None:
            skipped_no_rank += 1
            continue

        # Look up gold spans from the test set via PMCID (may be multiple)
        gold_spans = gold_spans_by_pmcid.get(pmcid)

        if not gold_spans:
            skipped_no_gold += 1
            continue

        # Build prompt
        user_msg = build_user_message(query, gold_spans, eval_span_item["text"])

        # Call LLM with retry
        raw_answer = None
        score = None

        for attempt in range(5):
            try:
                raw_answer = caller_fn(
                    args.api_key, api_model, SYSTEM_PROMPT, user_msg,
                    args.temperature, args.top_p,
                )
                score = parse_score(raw_answer)
                time.sleep(0.5)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    wait = min(32, 2 ** (attempt + 1))
                    print(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Error for {pmcid}: {e}")
                    errors += 1
                    break
        else:
            print(f"Rate limit exhausted for {pmcid}, skipping (will retry on resume)")
            errors += 1
            continue

        judgment = {
            "PMCID": pmcid,
            "query": query,
            "eval_rank": eval_rank,
            "retrieved_span": eval_span_item["text"],
            "retrieved_span_label": eval_span_item["label"],
            "gold_spans": gold_spans,
            "llm_messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "model": model_key,
            "llm_raw_answer": raw_answer,
            "llm_score": score,
        }

        judgments.append(judgment)

        if score is not None:
            scores.append(score)

        # Write continuously
        with open(judgments_path, "w") as f:
            json.dump(judgments, f, indent=2)

        current_mean = sum(scores) / len(scores) if scores else 0.0
        pbar.set_postfix(score=score, mean=f"{current_mean:.3f}", judged=len(judgments))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "input_file": str(args.input),
        "model": model_key,
        "eval_rank": eval_rank,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "total_entries": len(data),
        "judged_count": len(scores),
        "skipped_no_gold_in_top_k": skipped_no_gold,
        "skipped_no_span_at_rank": skipped_no_rank,
        "api_errors": errors,
        "mean_llm_score": round(mean_score, 4),
        "score_distribution": {
            str(s): scores.count(s) for s in range(4)
        },
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"Judged spans: {len(scores)}")
    print(f"Skipped (no gold in top-k): {skipped_no_gold}")
    print(f"Skipped (no span at rank): {skipped_no_rank}")
    print(f"API errors: {errors}")
    print(f"Mean LLM score: {mean_score:.4f}")
    print(f"Score distribution: {summary['score_distribution']}")
    print(f"\nJudgments saved to: {judgments_path}")
    print(f"Summary saved to: {summary_path}")


def main():
    """Parse CLI arguments and launch the evaluation."""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation of retrieval quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to retrieval results JSON file.",
    )
    parser.add_argument(
        "-g", "--gold-spans", default="test_gold_spans.jsonl",
        help="Path to the test gold spans JSONL file (grouped positives per PMCID).",
    )
    parser.add_argument(
        "-m", "--model", required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="LLM judge model to use.",
    )
    parser.add_argument(
        "-k", "--api-key", required=True,
        help="API key for the selected model provider.",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p", type=float, default=1,
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "-r", "--rank", type=int, default=1,
        help="Rank position of the retrieved document to evaluate.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="llm_judge_results",
        help="Directory for output files.",
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
