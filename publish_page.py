"""Write reviewed_page.html: the chart, its reviewed text alternative, and the data table.

The page follows W3C's figure-and-figcaption pattern for complex images: a short
alt text that identifies the chart, a visible long description, and a table with
<th> and <td> cells. It is a reviewed description, not a WCAG conformance claim.
"""

import json
from html import escape
from pathlib import Path

import checker

ROOT = Path(__file__).parent
RECORDED = ROOT / "recorded" / "recorded_outputs.json"
PAGE = ROOT / "reviewed_page.html"


def data_table(df) -> str:
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in ("python_pct", "sql_pct", "power_bi_pct"):
            value = row[col]
            cells.append("<td>No data</td>" if value != value else f"<td>{int(value)}%</td>")  # NaN != NaN
        rows.append(f"<tr><th scope=\"row\">Week {int(row['week'])}</th>{''.join(cells)}</tr>")
    return "\n".join(rows)


def build_page() -> str:
    reviewed = json.loads(RECORDED.read_text()).get("reviewed_description")
    if not reviewed:
        raise SystemExit("No reviewed_description in recorded_outputs.json. Review a passing run first.")
    paragraphs = "".join(f"<p>{escape(p)}</p>" for p in reviewed["long_description"].split("\n\n"))
    alt = escape(reviewed["alt_text"] + " Description and data table below.", quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LearnFlow completion rates</title>
<style>
  body {{ font-family: system-ui, sans-serif; color: #080808; background: #FFFFFF; max-width: 860px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }}
  img {{ width: 100%; height: auto; border: 1px solid #080808; }}
  table {{ border-collapse: collapse; margin-top: 0.5rem; }}
  th, td {{ border: 1px solid #080808; padding: 0.3rem 0.8rem; text-align: right; }}
  caption {{ text-align: left; font-weight: 600; padding-bottom: 0.4rem; }}
  .note {{ font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>LearnFlow course completion by weekly cohort</h1>
<figure role="group">
  <img src="learnflow_chart.png" alt="{alt}">
  <figcaption>
    <h2>Chart description</h2>
    {paragraphs}
    <table>
      <caption>Share of each weekly enrollment cohort that completed its course within 30 days</caption>
      <thead><tr><th scope="col">Cohort</th><th scope="col">Python</th><th scope="col">SQL</th><th scope="col">Power BI</th></tr></thead>
      <tbody>
{data_table(checker.load_data())}
      </tbody>
    </table>
  </figcaption>
</figure>
<p class="note">Description drafted by GPT-6.2 Sol from the chart image, checked against the source data, and edited by a person. Synthetic data.</p>
</body>
</html>
"""


if __name__ == "__main__":
    PAGE.write_text(build_page(), encoding="utf-8")
    print(f"wrote {PAGE}")
