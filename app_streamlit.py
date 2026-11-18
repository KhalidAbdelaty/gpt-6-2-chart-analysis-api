"""Streamlit front end for the LearnFlow Chart Explainer (GPT-6.2 Sol and Astra).

    streamlit run app_streamlit.py

"View the recorded run" replays recorded/recorded_outputs.json and never calls the API.
"Send one live request" is the only control that calls the OpenAI API; it is billed to the key in .env.
"""

from __future__ import annotations

import base64
import json
import os
import statistics
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import checker
from publish_page import build_page

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RECORDED = ROOT / "recorded" / "recorded_outputs.json"
INK, PAPER = "#080808", "#FFFFFF"  # the only two colors on the page

CONDITIONS = {
    "sol_image": "Sol, image only",
    "astra_image": "Astra, image only",
    "sol_image_data": "Sol, image + CSV",
    "astra_image_data": "Astra, image + CSV",
}

FEATURES = [
    (":material/image:", "Image input at detail high",
     "The same 1200 x 720 PNG goes to both models as an input_image item. Its SHA-256 hash is checked before every request."),
    (":material/tune:", "reasoning.effort = medium",
     "Set explicitly on every request, as the GPT-6.2 migration guide suggests for a new integration. No second effort is run."),
    (":material/forum:", "Developer message first",
     "Shared instructions sit in a developer message at the start of input, since GPT-6.2 appends top-level instructions after input."),
    (":material/data_object:", "Structured Outputs with Pydantic",
     "responses.parse() returns alt text, a long description, claims, and uncertainties in one fixed schema."),
    (":material/table_chart:", "Optional source CSV",
     "Conditions 3 and 4 add the CSV as a second input_text item after the image. That is the only intended change."),
    (":material/fact_check:", "Python check, then a person",
     "Every claim with a track, week, and value is checked against the CSV. A reviewer reads all 12 outputs and gives the verdict."),
]

st.set_page_config(page_title="LearnFlow Chart Explainer", page_icon=":material/bar_chart:", layout="wide")
if (ASSETS / "datacamp-logo.png").exists():
    st.logo(str(ASSETS / "datacamp-logo.png"), size="large")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1180px; }}
  .brand {{ display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; margin-bottom: .25rem; }}
  .brand img {{ height: 28px; width: auto; display: block; }}
  .pill {{ display: inline-block; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem;
          border: 1px solid {INK}; border-radius: 999px; padding: .15rem .65rem; color: {INK}; background: {PAPER}; }}
  .pill-solid {{ background: {INK}; color: {PAPER}; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .6rem; margin: .6rem 0 1rem; }}
  .stat {{ border: 1px solid {INK}; border-radius: .6rem; padding: .55rem .8rem; background: {PAPER}; }}
  .stat .label {{ font-size: .78rem; color: {INK}; }}
  .stat .value {{ font-size: 1.25rem; font-weight: 700; font-family: ui-monospace, Menlo, Consolas, monospace; color: {INK}; }}
  table.lf {{ border-collapse: collapse; width: 100%; margin: .3rem 0 .6rem; font-size: .9rem; }}
  table.lf th, table.lf td {{ border: 1px solid {INK}; padding: .35rem .6rem; text-align: left; vertical-align: top; color: {INK}; }}
  table.lf th {{ font-weight: 700; }}
  .textblock p {{ margin: 0 0 .6rem; }}
  code {{ color: {INK} !important; background: {PAPER} !important; border: 1px solid {INK}; border-radius: .3rem; }}
  table.lf tr, table.lf thead, table.lf tbody {{ border-color: {INK} !important; }}
  pre.prompt {{ white-space: pre-wrap; font-size: .82rem; margin: 0; color: {INK}; background: {PAPER}; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {INK} !important; opacity: 1 !important; }}
  [data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"] * {{ color: {INK} !important; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def stats_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(f"<div class='stat'><div class='label'>{escape(a)}</div>"
                    f"<div class='value'>{escape(b)}</div></div>" for a, b in items)
    return f"<div class='stats'>{cells}</div>"


def html_table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th scope='col'>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='lf'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def paragraphs(text: str) -> str:
    return "<div class='textblock'>" + "".join(f"<p>{escape(p)}</p>" for p in text.split("\n\n")) + "</div>"


@st.cache_data
def load_recorded() -> dict:
    return json.loads(RECORDED.read_text(encoding="utf-8"))


def condition_rows(records: list[dict]) -> tuple[list[list], list[list]]:
    quality, cost = [], []
    for cond, label in CONDITIONS.items():
        rs = [r for r in records if r["condition"] == cond]
        done = [r for r in rs if r.get("status") == "completed" and r.get("parsed")]
        usage = [r["usage"] for r in done]
        facts = [r.get("review", {}).get("required_facts_captured") for r in done]
        quality.append([
            label, f"{len(done)} of {len(rs)}",
            f"{sum(r.get('review', {}).get('passes', False) for r in done)} of {len(rs)}",
            f"{min(facts)} to {max(facts)} of 8" if facts else "n/a",
            sum(r.get("review", {}).get("unsupported_claims", 0) for r in done),
            f"{statistics.median(r['elapsed_s'] for r in done):.1f} s" if done else "n/a",
        ])
        cost.append([
            label,
            f"{round(statistics.mean(u['input_tokens'] for u in usage)):,}" if usage else "n/a",
            f"{sum(u['cached_tokens'] for u in usage):,}",
            f"{round(statistics.mean(u['output_tokens'] for u in usage)):,}" if usage else "n/a",
            round(statistics.mean(u["reasoning_tokens"] for u in usage)) if usage else "n/a",
            f"${statistics.mean(u['cost_usd'] for u in usage):.4f}" if usage else "n/a",
            f"${sum(u['cost_usd'] for u in usage):.4f}",
        ])
    return quality, cost


def show_run(rec: dict, df: pd.DataFrame) -> None:
    usage = rec.get("usage") or {}
    if usage:
        st.markdown(f"**{rec.get('returned_model')}** · status `{rec.get('status')}` · `store` {str(rec.get('store')).lower()}")
        st.markdown(stats_html([
            ("Time", f"{rec.get('elapsed_s')} s"),
            ("Input tokens", f"{usage['input_tokens']:,}"), ("Cached", f"{usage['cached_tokens']:,}"),
            ("Output tokens", f"{usage['output_tokens']:,}"), ("Reasoning", f"{usage['reasoning_tokens']:,}"),
            ("Cost", f"${usage['cost_usd']:.4f}"),
        ]), unsafe_allow_html=True)
    parsed = rec.get("parsed")
    if not parsed:
        with st.container(border=True):
            st.markdown(f"**No parsed output.** Status: `{rec.get('status')}`")
            st.markdown(escape(str(rec.get("refusal") or rec.get("error") or "")))
        return

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown(f"**Short alt text** ({len(parsed['alt_text'])} characters)")
            st.markdown(paragraphs(parsed["alt_text"]), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Long description**")
            st.markdown(paragraphs(parsed["long_description"]), unsafe_allow_html=True)
    with right:
        review = rec.get("review")
        with st.container(border=True):
            st.markdown("**Reviewed verdict**")
            if review:
                st.markdown(html_table(["Check", "Result"], [
                    ["Required facts", f"{review['required_facts_captured']} of 8"],
                    ["Unsupported claims", review["unsupported_claims"]],
                    ["Alt text 200 characters or fewer", "yes" if review["alt_text_ok"] else "no"],
                    ["Readability", review["readability"]],
                    ["Passes the rule", "yes" if review["passes"] else "no"],
                ]), unsafe_allow_html=True)
                if review.get("checker_flags_overruled"):
                    st.markdown(f"Checker flag overruled on review: {review['checker_flags_overruled']}")
            else:
                st.markdown("Not reviewed. This output came from a live request in this session.")
        with st.container(border=True):
            st.markdown("**Uncertainties the model listed**")
            for item in parsed["uncertainties"] or ["None"]:
                st.markdown(f"- {item}")

    score = rec.get("auto_score") or checker.score_run(df, parsed)
    missed = review.get("required_facts_missed", []) if review else None
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Required facts**")
        rows = [[key.split("_", 1)[0], text, "yes" if score["required_facts"][key] else "no",
                 "not reviewed" if missed is None else ("no" if key in missed else "yes")]
                for key, text in checker.REQUIRED_FACTS.items()]
        st.markdown(html_table(["ID", "Fact", "Python pattern", "Reviewed"], rows), unsafe_allow_html=True)
        st.caption("The pattern column only points the reviewer at a sentence. The reviewed column is the verdict.")
    with right, st.container(border=True):
        st.markdown("**Numeric claims checked against the CSV**")
        rows = [[n["source"].replace("_", " "), n["track"], n["week"], f"{n['value']:g}", n["verdict"]]
                for n in score["numeric_checks"]]
        st.markdown(html_table(["From", "Track", "Week", "Value", "Verdict"], rows), unsafe_allow_html=True)
        st.caption(f"Tolerance: {checker.TOLERANCE_PP:g} percentage points.")


# ----------------------------------------------------------------- header
df = checker.load_data()
data = load_recorded()
records = data["records"]

logo_svg = ASSETS / "openai-logo.svg"
logo = (f'<img src="data:image/svg+xml;base64,{base64.b64encode(logo_svg.read_bytes()).decode()}" alt="OpenAI logo"/>'
        if logo_svg.exists() else "")
st.markdown(f'<div class="brand">{logo}<span class="pill">gpt-6.2-sol</span>'
            f'<span class="pill">gpt-6.2-astra</span></div>', unsafe_allow_html=True)
st.title("LearnFlow Chart Explainer")
st.markdown("Can GPT-6.2 Sol and Astra describe the same chart accurately and accessibly, and does adding the "
            "source CSV change the answer? Both models write alt text and a long description; Python checks "
            "every number against the data, and a person gives the verdict.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Run settings")
    st.markdown(html_table(["Setting", "Value"], [
        ["Models", "gpt-6.2-sol, gpt-6.2-astra"], ["reasoning.effort", "medium"], ["Image detail", "high"],
        ["store", "false"], ["max_output_tokens", "25,000"], ["Runs", "3 per condition"],
        ["Recorded", data["test_date_utc"]], ["openai SDK", data["openai_sdk"]],
    ]), unsafe_allow_html=True)
    st.subheader("Standard price per 1M tokens")
    st.markdown(html_table(["", "Sol", "Astra"], [
        ["Input", "$2.00", "$8.00"], ["Cached input", "$0.10", "$0.40"],
        ["Cache write (30 min)", "$2.50", "$10.00"], ["Output", "$10.00", "$40.00"],
    ]), unsafe_allow_html=True)
    st.subheader("Live request")
    live_model = st.selectbox("Model", ["gpt-6.2-sol", "gpt-6.2-astra"])
    live_data = st.radio("Input", ["Image only", "Image + CSV"], horizontal=True) == "Image + CSV"
    if st.session_state.get("spend"):
        st.markdown(stats_html([("Spent this session", f"${st.session_state['spend']:.4f}")]),
                    unsafe_allow_html=True)

# --------------------------------------------------------------- features
st.subheader("What this run uses")
cols = st.columns(2)
for i, (icon, name, desc) in enumerate(FEATURES):
    with cols[i % 2], st.container(border=True):
        st.markdown(f"{icon} **{name}**")
        st.caption(desc)

# --------------------------------------------------------------- controls
st.write("")
left, right = st.columns(2)
with left:
    live = st.button("Send one live request", type="primary", icon=":material/send:", width="stretch",
                     help="Calls the OpenAI API once with the model and CSV choice from the sidebar.")
with right:
    if st.button("View the recorded run", icon=":material/history:", width="stretch",
                 help="Opens the 12 recorded requests without calling the API."):
        st.session_state["view"] = "recorded"
st.caption("A live request calls the OpenAI API and is billed to the key in your .env file. "
           "The recorded run makes no API call.")

if live:
    from dotenv import load_dotenv  # only needed when a request is about to be sent
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("No OPENAI_API_KEY found. Add it to a .env file next to app_streamlit.py.", icon=":material/key_off:")
    else:
        import describe  # the OpenAI client is created only here
        with st.status(f"Sending the chart to {live_model}...", expanded=False) as status:
            rec = describe.describe_chart(describe.make_client(), live_model,
                                          checker.CSV_PATH.read_text() if live_data else None)
            rec["condition"] = f"Live: {live_model}{' + CSV' if live_data else ', image only'}"
            rec["auto_score"] = checker.score_run(df, rec.get("parsed"))
            st.session_state.setdefault("live", []).append(rec)
            st.session_state["spend"] = st.session_state.get("spend", 0.0) + rec.get("usage", {}).get("cost_usd", 0.0)
            st.session_state["view"] = "live"
            status.update(label=f"{live_model}: {rec['status']}", state="complete")

# ---------------------------------------------------------------- results
st.divider()
live_runs = st.session_state.get("live", [])
if st.session_state.get("view") == "live" and live_runs:
    st.subheader("Live request")
    show_run(live_runs[-1], df)
    st.stop()

total = sum(r["usage"]["cost_usd"] for r in records if r.get("usage"))
st.subheader("Recorded run")
st.markdown(stats_html([
    ("Requests", str(len(records))),
    ("Completed", f"{sum(r.get('status') == 'completed' for r in records)} of {len(records)}"),
    ("Passed the rule", f"{sum(r.get('review', {}).get('passes', False) for r in records)} of {len(records)}"),
    ("Total spend", f"${total:.4f}"),
    ("Test date", data["test_date_utc"]),
]), unsafe_allow_html=True)

tabs = st.tabs([":material/bar_chart: Chart and data", ":material/description: Descriptions",
                ":material/leaderboard: Scorecard", ":material/public: Reviewed page", ":material/rule: Prompt and rubric"])

with tabs[0]:
    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        st.image(str(checker.PNG_PATH), width="stretch")
        st.caption("The only image sent to either model (1200 x 720 PNG). SQL has no point in week 8, "
                   "and the vertical axis starts at 50%.")
    with right, st.container(border=True):
        st.markdown("**Source data (percent)**")
        rows = [[f"Week {int(r.week)}", f"{int(r.python_pct)}%",
                 "No data" if pd.isna(r.sql_pct) else f"{int(r.sql_pct)}%", f"{int(r.power_bi_pct)}%"]
                for r in df.itertuples()]
        st.markdown(html_table(["Cohort", "Python", "SQL", "Power BI"], rows), unsafe_allow_html=True)

with tabs[1]:
    labels = {i: f"{CONDITIONS[r['condition']]}, run {r['run']}" for i, r in enumerate(records)}
    choice = st.selectbox("Output", list(labels), format_func=labels.get)
    show_run(records[choice], df)

with tabs[2]:
    quality, cost = condition_rows(records)
    with st.container(border=True):
        st.markdown("**Accuracy, 3 runs per condition**")
        st.markdown(html_table(["Condition", "Completed", "Passed", "Required facts", "Unsupported claims",
                                "Median time"], quality), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Tokens and cost, Standard rates**")
        st.markdown(html_table(["Condition", "Input (mean)", "Cached (3 runs)", "Output (mean)",
                                "Reasoning (mean)", "Mean cost", "Total cost"], cost), unsafe_allow_html=True)
        st.caption("Run 1 of each condition paid for a cache write; runs 2 and 3 read that prefix from cache.")

with tabs[3]:
    reviewed = data.get("reviewed_description")
    if not reviewed:
        st.markdown("This recording has no reviewed description yet.")
    else:
        st.caption(f"Source: {reviewed['source']}. A reviewed description, not a WCAG conformance claim.")
        left, right = st.columns([3, 2])
        with left, st.container(border=True):
            st.image(str(checker.PNG_PATH), width="stretch")
            st.markdown(paragraphs(reviewed["long_description"]), unsafe_allow_html=True)
        with right:
            with st.container(border=True):
                st.markdown("**alt attribute**")
                st.markdown(paragraphs(reviewed["alt_text"]), unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("**Edits made in review**")
                for edit in reviewed["edits"]:
                    st.markdown(f"- {edit}")
            st.download_button("Download reviewed_page.html", build_page(), file_name="reviewed_page.html",
                               mime="text/html", icon=":material/download:", width="stretch")

with tabs[4]:
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Developer message, identical in all 12 requests**")
        st.markdown("<pre class='prompt'>" + escape(data["developer_prompt"]).replace("\n", "<br>") + "</pre>",
                    unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("**Pass rule, fixed before the first request**")
            st.markdown(checker.PASS_RULE)
        with st.container(border=True):
            facts = checker.reference_facts(df)
            st.markdown("**Reference facts computed from the CSV**")
            st.markdown(html_table(["Fact", "Value"], [
                ["SQL week 1 to week 12", f"{facts['sql_week1']:g}% to {facts['sql_week12']:g}% "
                                          f"(+{facts['sql_gain_pp']:g} points)"],
                ["First week SQL is above Python", f"Week {facts['first_week_sql_above_python']}"],
                ["Missing SQL week", ", ".join(str(w) for w in facts["sql_missing_weeks"])],
                ["Vertical axis starts at", f"{facts['y_axis_floor']}%"],
                ["PNG SHA-256", data["png_sha256"][:16] + "..."],
            ]), unsafe_allow_html=True)
