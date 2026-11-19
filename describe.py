"""One function, one schema, one prompt: describe the LearnFlow chart with GPT-6.2.

Usage:
    python describe.py            # 4 conditions x 3 runs, writes recorded/live_outputs.json
    python describe.py --runs 1   # one run per condition

The only thing that changes between conditions is the model ID and whether the
CSV is added as a second text item after the image.
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import checker

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "recorded" / "live_outputs.json"

# Standard short-context rates per 1M tokens, checked on the pricing page on November 18, 2026.
# Cache writes use the 30-minute rate. Reasoning tokens are billed as output.
RATES = {
    "gpt-6.2-sol": {"input": 2.00, "cached": 0.10, "cache_write": 2.50, "output": 10.00},
    "gpt-6.2-astra": {"input": 8.00, "cached": 0.40, "cache_write": 10.00, "output": 40.00},
}

CONDITIONS = [
    {"id": "sol_image", "model": "gpt-6.2-sol", "with_data": False},
    {"id": "astra_image", "model": "gpt-6.2-astra", "with_data": False},
    {"id": "sol_image_data", "model": "gpt-6.2-sol", "with_data": True},
    {"id": "astra_image_data", "model": "gpt-6.2-astra", "with_data": True},
]

# Shared instructions go in a developer message at the start of input. From GPT-6.2,
# top-level `instructions` are appended after `input` and billed as uncached input.
DEVELOPER_PROMPT = """You write text alternatives for charts on LearnFlow course pages. Some readers use screen readers.

Return four fields:
- alt_text: one sentence of at most 200 characters that identifies the chart: its type, subject, and time span. Do not list values in it.
- long_description: one or two paragraphs for a reader who cannot see the chart. Cover the scales, values, relationships, and trends the chart shows, and anything a sighted reader would notice.
- claims: every factual statement in long_description, one per item. Fill track, week, and value_percent when a claim is about one track's value in one week; otherwise set them to null.
- uncertainties: anything you could not determine or had to estimate.

Rules:
- Describe only what the chart and any data you are given show. Do not explain why the numbers changed.
- When you read a value from the image instead of from supplied data, say it is approximate.
- Use percentage points for the difference between two percentages."""

TASK_TEXT = "Write the text alternative for this chart."


class Claim(BaseModel):
    statement: str
    track: Literal["Python", "SQL", "Power BI"] | None
    week: int | None
    value_percent: float | None


class ChartDescription(BaseModel):
    alt_text: str
    long_description: str
    claims: list[Claim]
    uncertainties: list[str]


def make_client() -> OpenAI:
    # The SDK reads OPENAI_API_KEY from the environment; load_dotenv puts .env there first.
    load_dotenv(ROOT / ".env")
    load_dotenv()  # also accept a .env in the current folder
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or your environment.")
    return OpenAI()


def image_data_url(path: Path = checker.PNG_PATH) -> str:
    if checker.file_sha256(path) != checker.PNG_SHA256:
        raise SystemExit("learnflow_chart.png changed since the rubric was frozen. Rebuild or restore it.")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def request_cost(model: str, usage) -> dict:
    rates = RATES[model]
    details = usage.input_tokens_details
    cached = details.cached_tokens or 0
    written = getattr(details, "cache_write_tokens", 0) or 0
    ordinary = usage.input_tokens - cached - written  # each input token has one rate
    usd = (ordinary * rates["input"] + cached * rates["cached"] + written * rates["cache_write"]
           + usage.output_tokens * rates["output"]) / 1_000_000
    return {"input_tokens": usage.input_tokens, "ordinary_input_tokens": ordinary,
            "cached_tokens": cached, "cache_write_tokens": written,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
            "cost_usd": round(usd, 6)}


def describe_chart(client: OpenAI, model: str, source_csv: str | None = None) -> dict:
    """Send the frozen chart (and, optionally, its CSV) to one model and return a record."""
    content = [
        {"type": "input_text", "text": TASK_TEXT},
        {"type": "input_image", "image_url": image_data_url(), "detail": "high"},
    ]
    if source_csv is not None:
        # The only intended change between conditions: the source table, after the image.
        content.append({"type": "input_text",
                        "text": "Source data used to draw this chart (CSV, values in percent):\n\n" + source_csv})

    record = {"model": model, "with_data": source_csv is not None, "store": False,
              "reasoning_effort": "medium", "detail": "high",
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    start = time.perf_counter()
    try:
        response = client.responses.parse(
            model=model,
            input=[{"role": "developer", "content": DEVELOPER_PROMPT},
                   {"role": "user", "content": content}],
            reasoning={"effort": "medium"},
            text_format=ChartDescription,
            max_output_tokens=25_000,
            store=False,
        )
    except openai.APIError as err:
        record.update(status="api_error", error=f"{type(err).__name__}: {err}",
                      elapsed_s=round(time.perf_counter() - start, 2))
        return record
    except Exception as err:  # e.g. a truncated object that fails validation
        record.update(status="parse_error", error=f"{type(err).__name__}: {err}",
                      elapsed_s=round(time.perf_counter() - start, 2))
        return record
    record["elapsed_s"] = round(time.perf_counter() - start, 2)

    refusal = None
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    refusal = part.refusal
    record.update(
        response_id=response.id,
        returned_model=response.model,
        service_tier=getattr(response, "service_tier", None),
        status=response.status,
        incomplete_reason=getattr(response.incomplete_details, "reason", None),
        refusal=refusal,
        usage=request_cost(model, response.usage),
    )
    # Check status and refusal before trusting output_parsed.
    if response.status == "completed" and refusal is None and response.output_parsed is not None:
        record["parsed"] = response.output_parsed.model_dump()
    else:
        record["parsed"] = None
        record["raw_text"] = response.output_text
    return record


def run_experiment(runs: int) -> dict:
    if checker.file_sha256(checker.PNG_PATH) != checker.PNG_SHA256:
        raise SystemExit("learnflow_chart.png changed since the rubric was frozen. Rebuild or restore it.")
    if checker.csv_sha256(checker.CSV_PATH) != checker.CSV_SHA256:
        raise SystemExit("learnflow_completion.csv changed since the rubric was frozen. Restore it before running.")
    client = make_client()
    df = checker.load_data()
    csv_text = checker.CSV_PATH.read_text()
    records = []
    for run in range(1, runs + 1):
        for cond in CONDITIONS:  # interleave conditions so no model gets one time slot
            rec = describe_chart(client, cond["model"], csv_text if cond["with_data"] else None)
            rec.update(condition=cond["id"], run=run)
            rec["auto_score"] = checker.score_run(df, rec.get("parsed"))
            cost = rec.get("usage", {}).get("cost_usd")
            print(f"{cond['id']:<17} run {run}  {rec['status']:<10} {rec.get('elapsed_s')} s  ${cost}")
            records.append(rec)
    return {
        "test_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "openai_sdk": openai.__version__,
        "png_sha256": checker.PNG_SHA256,
        "csv_sha256": checker.csv_sha256(checker.CSV_PATH),
        "developer_prompt": DEVELOPER_PROMPT,
        "task_text": TASK_TEXT,
        "schema": ChartDescription.model_json_schema(),
        "pass_rule": checker.PASS_RULE,
        "required_facts": checker.REQUIRED_FACTS,
        "tolerance_pp": checker.TOLERANCE_PP,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    results = run_experiment(args.runs)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"saved {args.out}")
