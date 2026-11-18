"""Reference facts, scoring rules, and the pass rule for the LearnFlow test.

Everything in this file was fixed before the first API call. Nothing here is
sent to a model: the image-only prompts never include the CSV, and no prompt
includes these facts or this checklist.

Python checks what it can check mechanically (numbers against the CSV, plus
pattern flags for the eight required facts). A person reads every output, and
the reviewed verdict is the one that counts. Readability is a human judgment
and is recorded separately from the pass rule.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "learnflow_completion.csv"
PNG_PATH = ROOT / "learnflow_chart.png"

# Frozen fixture fingerprints. describe.py refuses to run if either changes.
PNG_SHA256 = "33c34cf6b207e1cd7b1d2438bd8ffa8611cb5d99c45c04a54f275be0af29ed60"
Y_AXIS_FLOOR = 50  # matches Y_MIN in build_chart.py

# A value read from the image counts as correct within 2 percentage points.
# Gridlines sit 5 points apart and the chart prints no data labels.
TOLERANCE_PP = 2.0

TRACKS = {"Python": "python_pct", "SQL": "sql_pct", "Power BI": "power_bi_pct"}

REQUIRED_FACTS = {
    "R1_identity": "Identifies a line chart of 30-day completion rates for Python, SQL, and Power BI over 12 weekly cohorts",
    "R2_sql_rises": "SQL rises across the period",
    "R3_sql_endpoints": "SQL starts near 58% (week 1) and ends near 81% (week 12), within 2 points",
    "R4_crossover": "SQL first moves above Python in week 11 ('between weeks 10 and 11' also counts)",
    "R5_week8_missing": "SQL has no value in week 8, described as missing data, not as zero or a drop",
    "R6_axis_floor": "The vertical axis starts at 50%, not 0%",
    "R7_python_flat": "Python stays in a narrow band (about 72% to 78%) and eases slightly at the end",
    "R8_powerbi_declines": "Power BI drifts down, from about 66% to about 60%",
}

PASS_RULE = (
    "A run passes only if (1) the response completed and parsed with no refusal, "
    "(2) all 8 required facts appear in the alt text or long description, "
    "(3) it makes 0 unsupported claims, and "
    "(4) the alt text is 200 characters or fewer and names the chart type and subject. "
    "Unsupported claims are: a value off by more than 2 points, any value for SQL in week 8, "
    "treating week 8 as zero or a drop, a crossover in any week other than 11, "
    "an axis that starts anywhere but 50%, and any cause for the trends."
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path: Path = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def reference_facts(df: pd.DataFrame) -> dict:
    """Compute the answer key from the CSV and confirm the brief's three traps hold."""
    sql, py = df["sql_pct"], df["python_pct"]
    both = df.dropna(subset=["sql_pct"])
    sql_ahead = both.loc[both["sql_pct"] > both["python_pct"], "week"].tolist()
    facts = {
        "weeks": int(df["week"].max()),
        "sql_week1": float(sql.iloc[0]),
        "sql_week12": float(sql.iloc[-1]),
        "sql_gain_pp": float(sql.iloc[-1] - sql.iloc[0]),
        "sql_missing_weeks": df.loc[sql.isna(), "week"].tolist(),
        "first_week_sql_above_python": sql_ahead[0] if sql_ahead else None,
        "weeks_sql_above_python": sql_ahead,
        "python_range": (float(py.min()), float(py.max())),
        "power_bi_week1": float(df["power_bi_pct"].iloc[0]),
        "power_bi_week12": float(df["power_bi_pct"].iloc[-1]),
        "y_axis_floor": Y_AXIS_FLOOR,
    }
    # The brief's facts must hold in the frozen data, or the test is broken.
    assert facts["sql_week1"] == 58 and facts["sql_week12"] == 81
    assert facts["sql_gain_pp"] == 23
    assert facts["sql_missing_weeks"] == [8]
    assert facts["weeks_sql_above_python"] == [11, 12]  # no earlier lead to confuse the crossover
    return facts


# ---------- numeric checks ----------

def true_value(df: pd.DataFrame, track: str, week: int):
    row = df.loc[df["week"] == week]
    if row.empty or track not in TRACKS:
        return None
    value = row[TRACKS[track]].iloc[0]
    return None if pd.isna(value) else float(value)


def judge_value(df: pd.DataFrame, track: str, week: int, value: float) -> str:
    """Return exact, within_tolerance, wrong, invented_missing, or not_checkable."""
    if track not in TRACKS or week is None or not 1 <= int(week) <= 12:
        return "not_checkable"
    truth = true_value(df, track, int(week))
    if truth is None:
        return "invented_missing"  # any value for SQL in week 8 is unsupported
    gap = abs(float(value) - truth)
    if gap == 0:
        return "exact"
    return "within_tolerance" if gap <= TOLERANCE_PP else "wrong"


PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?(?:%|percent\b)(?!\s*(?:points?|pp)\b)")
WEEK = re.compile(r"\bweeks?\s+(\d{1,2})\b")
SENTENCE = re.compile(r"(?<=[.;!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text or "") if s.strip()]


def tracks_in(sentence: str) -> list[str]:
    found = []
    low = sentence.lower()
    if "python" in low:
        found.append("Python")
    if re.search(r"\bsql\b", low):
        found.append("SQL")
    if "power bi" in low:
        found.append("Power BI")
    return found


def prose_values(text: str) -> list[dict]:
    """Pull (track, week, value) triples from sentences that name exactly one track.

    A sentence like 'SQL rises from about 58% in week 1 to 81% in week 12' pairs
    weeks and percentages in order. Sentences that mix tracks are left to the reviewer.
    """
    triples = []
    for s in sentences(text):
        names = tracks_in(s)
        weeks = [int(w) for w in WEEK.findall(s.lower())]
        values = [float(v) for v in PCT.findall(s)]
        values = [v for v in values if v != Y_AXIS_FLOOR or "axis" not in s.lower()]
        if len(names) == 1 and weeks and len(weeks) == len(values):
            for week, value in zip(weeks, values):
                triples.append({"track": names[0], "week": week, "value": value, "sentence": s})
    return triples


def check_numbers(df: pd.DataFrame, parsed: dict) -> list[dict]:
    """Check every numeric claim in the claims list and in the long description."""
    results = []
    for claim in parsed.get("claims", []):
        if claim.get("value_percent") is not None and claim.get("week") is not None:
            verdict = judge_value(df, claim.get("track"), claim["week"], claim["value_percent"])
            results.append({"source": "claims", "track": claim.get("track"), "week": claim["week"],
                            "value": claim["value_percent"], "verdict": verdict,
                            "text": claim.get("statement", "")})
    for t in prose_values(parsed.get("long_description", "")):
        verdict = judge_value(df, t["track"], t["week"], t["value"])
        results.append({"source": "long_description", "track": t["track"], "week": t["week"],
                        "value": t["value"], "verdict": verdict, "text": t["sentence"]})
    return results


# ---------- pattern flags for the required facts ----------
# These flags point the reviewer at the right sentence. They are not the verdict.

def _any_sentence(text: str, must: list[str], any_of: list[str]) -> str | None:
    for s in sentences(text):
        low = s.lower()
        if all(re.search(m, low) for m in must) and any(re.search(a, low) for a in any_of):
            return s
    return None


def flag_required_facts(df: pd.DataFrame, parsed: dict) -> dict:
    text = f"{parsed.get('alt_text', '')} {parsed.get('long_description', '')}"
    low = text.lower()
    flags = {}

    ident = all(k in low for k in ("python", "power bi")) and re.search(r"\bsql\b", low)
    ident = ident and "line" in low and "complet" in low
    ident = ident and re.search(r"12 (weekly )?(weeks|cohorts)|weeks? 1\D{1,6}12|twelve|week 12", low)
    flags["R1_identity"] = bool(ident)

    flags["R2_sql_rises"] = bool(_any_sentence(text, [r"\bsql\b"],
                                 [r"\brise", r"\brose\b", r"increas", r"climb", r"\bgrow", r"upward", r"\bgain", r"improv"]))

    prose = check_numbers(df, {"long_description": text, "claims": []})
    sql_vals = [n for n in prose if n["track"] == "SQL"]
    ok = {n["week"] for n in sql_vals if n["verdict"] in ("exact", "within_tolerance")}
    flags["R3_sql_endpoints"] = {1, 12} <= ok

    flags["R4_crossover"] = bool(_any_sentence(
        text, [r"\bsql\b", r"python"],
        [r"week 11", r"weeks? 10 and 11", r"week 10 and week 11", r"weeks? 10\D{1,4}11"]))

    flags["R5_week8_missing"] = bool(_any_sentence(
        text, [r"week 8"],
        [r"missing", r"no data", r"\bgap", r"break", r"broken", r"no (value|point|marker|observation|reading)",
         r"not (shown|plotted|recorded|reported|available)", r"absent", r"unavailable", r"blank", r"omitted"]))

    flags["R6_axis_floor"] = bool(re.search(
        r"(axis|scale|baseline|floor|starts?|begins?|range|lower bound)[^.]{0,60}\b50\s?%?"
        r"|\b50\s?%?[^.]{0,60}(axis|baseline|floor|lower bound)", low))

    flags["R7_python_flat"] = bool(_any_sentence(
        text, [r"python"],
        [r"stable", r"steady", r"flat", r"level", r"plateau", r"narrow", r"little", r"modest", r"slight",
         r"gradual", r"gentl", r"edges?", r"eases?", r"holds?", r"hover", r"peak", r"7\d\s?%?\s?(to|and|-)\s?7\d"]))

    flags["R8_powerbi_declines"] = bool(_any_sentence(
        text, [r"power bi"],
        [r"declin", r"drift", r"\bfall", r"\bfell\b", r"decreas", r"downward", r"edges? down", r"slip",
         r"\bdrops?\b", r"dipp", r"lower"]))
    return flags


# ---------- pattern flags for unsupported claims ----------

CAUSAL = re.compile(r"\b(because|due to|driven by|caused by|as a result of|thanks to|owing to|"
                    r"attribut\w+ to|result(s|ed)? from|reflect(s|ing)? (an? )?(improv|change|demand|interest))\b")


def flag_unsupported(df: pd.DataFrame, parsed: dict) -> list[dict]:
    text = f"{parsed.get('alt_text', '')} {parsed.get('long_description', '')}"
    flags = []
    for n in check_numbers(df, parsed):
        if n["verdict"] in ("wrong", "invented_missing"):
            flags.append({"type": n["verdict"], "text": n["text"]})
    for s in sentences(text):
        low = s.lower()
        if CAUSAL.search(low):
            flags.append({"type": "causal", "text": s})
        if "week 8" in low and re.search(r"\b0\s?%|zero|drops? to|falls? to", low):
            flags.append({"type": "week8_as_value", "text": s})
        if (re.search(r"\bsql\b", low) and "python" in low
                and re.search(r"overt(ake|ook)|surpass|pass(es|ed)?\b|cross", low)
                and re.search(r"week (9|10|12)\b", low) and not re.search(r"week 11|10 and 11|10\D{1,4}11", low)):
            flags.append({"type": "crossover_week", "text": s})
        if re.search(r"(axis|scale)[^.]{0,40}\b(0|zero)\s?%", low):
            flags.append({"type": "axis_zero", "text": s})
    return flags


def score_run(df: pd.DataFrame, parsed: dict | None) -> dict:
    """Python's half of the score. The reviewer confirms or corrects it."""
    if not parsed:
        return {"parsed": False}
    facts = flag_required_facts(df, parsed)
    numbers = check_numbers(df, parsed)
    return {
        "parsed": True,
        "required_facts": facts,
        "required_captured": sum(facts.values()),
        "numeric_checks": numbers,
        "numeric_exact": sum(n["verdict"] == "exact" for n in numbers),
        "numeric_within_tolerance": sum(n["verdict"] == "within_tolerance" for n in numbers),
        "numeric_wrong": sum(n["verdict"] in ("wrong", "invented_missing") for n in numbers),
        "unsupported_flags": flag_unsupported(df, parsed),
        "alt_text_chars": len(parsed.get("alt_text", "")),
    }


if __name__ == "__main__":
    data = load_data()
    print("PNG hash matches:", file_sha256(PNG_PATH) == PNG_SHA256)
    for key, value in reference_facts(data).items():
        print(f"{key}: {value}")
    print("\nPass rule:", PASS_RULE)
