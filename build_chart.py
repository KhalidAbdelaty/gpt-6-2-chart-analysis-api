"""Draw the LearnFlow completion chart from the frozen CSV.

The PNG this script writes is the only image sent to the models. Run it once,
then leave the file alone: describe.py refuses to call the API if the PNG hash
no longer matches the one recorded in checker.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, and the output stays byte-stable
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "learnflow_completion.csv"
PNG_PATH = ROOT / "learnflow_chart.png"

# Frozen display settings. The axis floor is part of the test, so it lives here.
Y_MIN, Y_MAX, Y_STEP = 50, 90, 5
TRACKS = [
    # column, label, color, marker
    ("python_pct", "Python", "#1F5FAD", "o"),
    ("sql_pct", "SQL", "#C2410C", "s"),
    ("power_bi_pct", "Power BI", "#2F7D32", "^"),
]


def build_chart(csv_path: Path = CSV_PATH, png_path: Path = PNG_PATH) -> Path:
    df = pd.read_csv(csv_path)  # the blank SQL cell in week 8 becomes NaN

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for column, label, color, marker in TRACKS:
        # NaN stays in the series, so Matplotlib breaks the line at week 8.
        # Dropping the row would draw a bridge; filling it with 0 would draw a dip.
        ax.plot(df["week"], df[column], color=color, marker=marker,
                markersize=7, linewidth=2.4, label=label)
        last = df[column].iloc[-1]
        # Direct labels at the line ends, so color is not the only cue.
        ax.annotate(label, xy=(12, last), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=12, fontweight="bold", color=color)

    ax.set_title("LearnFlow: share of each weekly cohort that completed its course within 30 days",
                 fontsize=13, loc="left", pad=12)
    ax.set_xlabel("Enrollment cohort week", fontsize=12)
    ax.set_ylabel("Completion rate (%)", fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.6, 12.9)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(range(Y_MIN, Y_MAX + 1, Y_STEP))
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    # Empty metadata keeps the bytes identical between rebuilds.
    fig.savefig(png_path, metadata={"Software": None})
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    path = build_chart()
    print(f"wrote {path}")
