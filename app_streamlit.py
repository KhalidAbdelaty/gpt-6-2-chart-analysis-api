"""LearnFlow Chart Explainer: replay the recorded GPT-6.2 run, or send one live request.

    streamlit run app_streamlit.py

Recorded mode reads recorded/recorded_outputs.json and never calls the API.
Live mode needs OPENAI_API_KEY in .env and sends exactly one request per click.
"""

import json
import os
import statistics
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import checker
from publish_page import build_page

ROOT = Path(__file__).parent
RECORDED = ROOT / "recorded" / "recorded_outputs.json"
INK, PAPER = "#080808", "#FFFFFF"  # the only two colors in the interface

CONDITION_LABELS = {
    "sol_image": "Sol, image only",
    "astra_image": "Astra, image only",
    "sol_image_data": "Sol, image + CSV",
    "astra_image_data": "Astra, image + CSV",
}

st.set_page_config(page_title="LearnFlow Chart Explainer", layout="wide")
st.markdown(f"""
<style>
  html, body, .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {{ background: {PAPER}; color: {INK}; }}
  .stApp *, [data-testid="stSidebar"] * {{ color: {INK}; border-color: {INK} !important; }}
  a, a * {{ color: {INK} !important; text-decoration: underline; }}
  code {{ background: {PAPER} !important; color: {INK} !important; border: 1px solid {INK}; }}
  [data-testid="stImageCaption"], [data-testid="stCaptionContainer"] {{ color: {INK} !important; opacity: 1 !important; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid {INK}; }}
  [data-testid="stTab"], [data-testid="stTab"] * {{ color: {INK} !important; font-weight: 600; }}
  [data-testid="stTab"] .react-aria-SelectionIndicator {{ background: {INK} !important; height: 3px; }}
  [data-testid="stRadioOption"] > div > div:first-child {{ background: {PAPER} !important; border: 2px solid {INK} !important; }}
  [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {{ background: {INK} !important; }}
  [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div {{ background: {PAPER} !important; }}
  .stButton button, .stDownloadButton button {{ background: {INK} !important; border: 1px solid {INK}; }}
  .stButton button p, .stDownloadButton button p {{ color: {PAPER} !important; }}
  .stButton button:hover, .stDownloadButton button:hover {{ background: {PAPER} !important; }}
  .stButton button:hover p, .stDownloadButton button:hover p {{ color: {INK} !important; }}
  .stButton button:disabled {{ background: {PAPER} !important; border: 1px dashed {INK}; }}
  .stButton button:disabled p {{ color: {INK} !important; }}
  [data-testid="stSelectbox"] div, [data-baseweb="select"] > div, [data-baseweb="popover"] *, [role="listbox"], [role="option"] {{ background: {PAPER} !important; }}
  [data-testid="stCheckbox"] label > span:first-child, [data-testid="stCheckbox"] div:first-child {{ background-color: {PAPER}; }}
  [data-testid="stExpander"] details, [data-testid="stExpander"] summary {{ background: {PAPER} !important; }}
  table.lf {{ border-collapse: collapse; width: 100%; margin: 0.4rem 0 1rem 0; }}
  table.lf th, table.lf td {{ border: 1px solid {INK}; padding: 0.35rem 0.6rem; text-align: left; vertical-align: top; }}
  table.lf th {{ font-weight: 700; }}
  .box {{ border: 2px solid {INK}; padding: 0.8rem 1rem; margin-bottom: 0.8rem; }}
  .label {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
  pre.prompt {{ border: 1px solid {INK}; background: {PAPER}; padding: 0.8rem; white-space: pre-wrap; font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)


def html_table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th scope='col'>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='lf'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


@st.cache_data
def load_recorded() -> dict:
    return json.loads(RECORDED.read_text())


def show_description(rec: dict, df: pd.DataFrame) -> None:
    parsed = rec.get("parsed")
    if not parsed:
        st.markdown(f"<div class='box'>No parsed output. Status: {escape(str(rec.get('status')))}. "
                    f"{escape(str(rec.get('refusal') or rec.get('error') or ''))}</div>", unsafe_allow_html=True)
        return
    st.markdown(f"<div class='box'><div class='label'>Short alt text ({len(parsed['alt_text'])} characters)</div>"
                f"{escape(parsed['alt_text'])}</div>", unsafe_allow_html=True)
    paragraphs = "".join(f"<p>{escape(p)}</p>" for p in parsed["long_description"].split("\n\n"))
    st.markdown(f"<div class='box'><div class='label'>Long description</div>{paragraphs}</div>",
                unsafe_allow_html=True)
    if parsed["uncertainties"]:
        items = "".join(f"<li>{escape(u)}</li>" for u in parsed["uncertainties"])
        st.markdown(f"<div class='box'><div class='label'>Uncertainties the model listed</div><ul>{items}</ul></div>",
                    unsafe_allow_html=True)

    score = rec.get("auto_score") or checker.score_run(df, parsed)
    review = rec.get("review", {})
    left, right = st.columns(2)
    with left:
        st.markdown("**Required facts**")
        missed = review.get("required_facts_missed", []) if review else None
        rows = [[key.split("_", 1)[0], text, "yes" if score["required_facts"][key] else "no",
                 "not reviewed" if missed is None else ("no" if key in missed else "yes")]
                for key, text in checker.REQUIRED_FACTS.items()]
        st.markdown(html_table(["ID", "Fact", "Python pattern", "Reviewed"], rows), unsafe_allow_html=True)
        st.markdown("The pattern flags only point the reviewer at a sentence; the reviewed column is the verdict.")
        if review:
            st.markdown(f"Reviewed: **{review['required_facts_captured']} of 8 captured**, "
                        f"**{review['unsupported_claims']} unsupported claims**, readability: {review['readability']}. "
                        f"Pass: **{'yes' if review['passes'] else 'no'}**.")
            if review.get("checker_flags_overruled"):
                st.markdown(f"Checker flag overruled on review: {review['checker_flags_overruled']}")
    with right:
        st.markdown("**Numeric claims checked against the CSV** (tolerance 2 points)")
        rows = [[n["source"], n["track"], n["week"], n["value"], n["verdict"]] for n in score["numeric_checks"]]
        st.markdown(html_table(["From", "Track", "Week", "Value", "Verdict"], rows), unsafe_allow_html=True)


def scorecard(records: list[dict]) -> list[list]:
    rows = []
    for cond, label in CONDITION_LABELS.items():
        rs = [r for r in records if r["condition"] == cond]
        done = [r for r in rs if r.get("status") == "completed" and r.get("parsed")]
        usage = [r["usage"] for r in done]
        passed = sum(r.get("review", {}).get("passes", False) for r in done)
        facts = [r.get("review", {}).get("required_facts_captured") for r in done]
        cost = [u["cost_usd"] for u in usage]
        rows.append([
            label, f"{len(done)} of {len(rs)}", f"{passed} of {len(rs)}",
            f"{min(facts)} to {max(facts)} of 8" if facts else "n/a",
            sum(r.get("review", {}).get("unsupported_claims", 0) for r in done),
            f"{statistics.median(r['elapsed_s'] for r in done):.1f} s" if done else "n/a",
            round(statistics.mean(u["input_tokens"] for u in usage)) if usage else "n/a",
            sum(u["cached_tokens"] for u in usage),
            round(statistics.mean(u["output_tokens"] for u in usage)) if usage else "n/a",
            round(statistics.mean(u["reasoning_tokens"] for u in usage)) if usage else "n/a",
            f"${statistics.mean(cost):.4f}" if cost else "n/a",
            f"${sum(cost):.4f}",
        ])
    return rows


# ---------- page ----------
logo_a, logo_b, _ = st.columns([1, 1, 4], vertical_alignment="center")
logo_a.image(str(ROOT / "assets" / "openai-logo.png"), width=140)
logo_b.image(str(ROOT / "assets" / "datacamp-logo.png"), width=170)
st.title("LearnFlow Chart Explainer")
st.markdown("One chart, two GPT-6.2 models, with and without the source CSV. "
            "Every description is checked against the data before it can be published.")

df = checker.load_data()
data = load_recorded()
records = data["records"]

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("Mode", ["Recorded run (no API calls)", "Live request"], label_visibility="collapsed")
    if mode.startswith("Recorded"):
        st.markdown(f"Replaying `recorded/recorded_outputs.json` from {data['test_date_utc']}, "
                    f"`openai` {data['openai_sdk']}. Nothing is sent to OpenAI.")
    else:
        from dotenv import load_dotenv  # imported only when a live request is possible
        load_dotenv(ROOT / ".env")
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        st.markdown("API key: **found in the environment**" if has_key
                    else "API key: **not set**. Add `OPENAI_API_KEY` to `.env`.")
        live_model = st.selectbox("Model", ["gpt-6.2-sol", "gpt-6.2-astra"])
        live_data = st.checkbox("Add the CSV after the image")
        st.markdown("One click sends one request at `medium` effort, `detail: high`. "
                    "Recorded runs cost $0.013 to $0.057 each.")
        if st.button("Send one request", disabled=not has_key):
            import describe  # the OpenAI client is created only here
            with st.spinner("Waiting for the model..."):
                rec = describe.describe_chart(describe.make_client(), live_model,
                                              checker.CSV_PATH.read_text() if live_data else None)
            rec["condition"] = f"live: {live_model}{' + CSV' if live_data else ''}"
            rec["auto_score"] = checker.score_run(df, rec.get("parsed"))
            st.session_state.setdefault("live", []).append(rec)

tab_chart, tab_desc, tab_score, tab_page = st.tabs(
    ["Chart and data", "Descriptions", "Scorecard", "Reviewed page"])

with tab_chart:
    col_img, col_tbl = st.columns([3, 2])
    with col_img:
        st.image(str(checker.PNG_PATH), caption="The only image sent to either model (1200 x 720 PNG)")
    with col_tbl:
        rows = [[f"Week {int(r.week)}", f"{int(r.python_pct)}%",
                 "No data" if pd.isna(r.sql_pct) else f"{int(r.sql_pct)}%", f"{int(r.power_bi_pct)}%"]
                for r in df.itertuples()]
        st.markdown(html_table(["Cohort", "Python", "SQL", "Power BI"], rows), unsafe_allow_html=True)
    facts = checker.reference_facts(df)
    settings = [
        ["Models", "gpt-6.2-sol, gpt-6.2-astra (Responses API)"],
        ["Settings held fixed", "reasoning.effort = medium, detail = high, store = false, one Pydantic schema"],
        ["PNG SHA-256", data["png_sha256"][:16] + "..."],
        ["Tolerance for values", f"{checker.TOLERANCE_PP:g} percentage points"],
        ["SQL week 1 to week 12", f"{facts['sql_week1']:g}% to {facts['sql_week12']:g}% "
                                  f"(+{facts['sql_gain_pp']:g} points)"],
        ["First week SQL is above Python", f"Week {facts['first_week_sql_above_python']}"],
        ["Missing SQL week", ", ".join(str(w) for w in facts["sql_missing_weeks"])],
        ["Vertical axis", f"starts at {facts['y_axis_floor']}%"],
    ]
    st.markdown("**Frozen before the first API call**")
    st.markdown(html_table(["Item", "Value"], settings), unsafe_allow_html=True)

with tab_desc:
    pool = records + st.session_state.get("live", [])
    labels = [f"{CONDITION_LABELS[r['condition']]}, run {r['run']}" if "run" in r
              else f"{r['condition']} (this session)" for r in pool]
    choice = st.selectbox("Output", range(len(pool)), format_func=lambda i: labels[i])
    rec = pool[choice]
    usage = rec.get("usage", {})
    if usage:
        st.markdown(html_table(
            ["Model returned", "Status", "Time", "Input", "Cached", "Cache write", "Output", "Reasoning", "Cost"],
            [[rec.get("returned_model"), rec.get("status"), f"{rec.get('elapsed_s')} s", usage["input_tokens"],
              usage["cached_tokens"], usage["cache_write_tokens"], usage["output_tokens"],
              usage["reasoning_tokens"], f"${usage['cost_usd']:.4f}"]]), unsafe_allow_html=True)
    show_description(rec, df)
    with st.expander("Developer message sent with every request"):
        st.markdown("<pre class='prompt'>" + escape(data["developer_prompt"]).replace("\n", "<br>") + "</pre>",
                    unsafe_allow_html=True)

with tab_score:
    st.markdown(f"**Pass rule (fixed before the run):** {checker.PASS_RULE}")
    st.markdown(html_table(
        ["Condition", "Completed", "Passed (reviewed)", "Required facts (reviewed)", "Unsupported", "Median time",
         "Input tokens", "Cached tokens (3 runs)", "Output tokens", "Reasoning tokens",
         "Mean cost", "Total cost"], scorecard(records)), unsafe_allow_html=True)
    total = sum(r["usage"]["cost_usd"] for r in records if r.get("usage"))
    st.markdown(f"Total spend for all 12 requests: **${total:.4f}** at Standard rates. Run 1 of each condition "
                "paid for a cache write; runs 2 and 3 read that prefix from cache.")

with tab_page:
    reviewed = data.get("reviewed_description")
    if not reviewed:
        st.markdown("This recording has no reviewed description yet. Review a passing run, then add "
                    "`reviewed_description` to the JSON file.")
        st.stop()
    st.markdown(f"Source: {reviewed['source']}. Reviewed by a person; not a WCAG conformance claim.")
    col_img, col_txt = st.columns([3, 2])
    with col_img:
        st.image(str(checker.PNG_PATH))
    with col_txt:
        st.markdown(f"<div class='box'><div class='label'>alt attribute</div>{escape(reviewed['alt_text'])}</div>",
                    unsafe_allow_html=True)
        edits = "".join(f"<li>{escape(e)}</li>" for e in reviewed["edits"])
        st.markdown(f"<div class='box'><div class='label'>Edits made in review</div><ul>{edits}</ul></div>",
                    unsafe_allow_html=True)
    paragraphs = "".join(f"<p>{escape(p)}</p>" for p in reviewed["long_description"].split("\n\n"))
    st.markdown(f"<div class='box'><div class='label'>Long description (visible on the page)</div>{paragraphs}</div>",
                unsafe_allow_html=True)
    st.download_button("Download reviewed_page.html", build_page(), file_name="reviewed_page.html",
                       mime="text/html")
