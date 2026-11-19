# LearnFlow Chart Explainer (GPT-6.2 Sol and Astra)

One synthetic chart, two models, two inputs. `describe.py` asks `gpt-6.2-sol` and `gpt-6.2-astra` for an accessible description of the same PNG, with and without the source CSV, and `checker.py` scores every answer against facts computed from that CSV before any request.

Tested on November 18, 2026 with `openai` 3.55.1 and Python 3.13, Responses API, `reasoning.effort` = `medium`, `detail` = `high`, `store` = `false`.

## Files

- `data/learnflow_completion.csv`: 12 weekly cohorts, three tracks, SQL blank in week 8
- `build_chart.py`: draws `learnflow_chart.png` (the only image sent to the models)
- `describe.py`: the one request function, the Pydantic schema, the developer prompt, and the 4 x 3 run loop
- `checker.py`: reference facts, the 8 required facts, the 2-point tolerance, and the pass rule
- `publish_page.py`: writes `reviewed_page.html` (chart, reviewed description, data table)
- `recorded/recorded_outputs.json`: the 12 recorded outputs with usage, cost, scores, and review
- `app_streamlit.py`: replays the recorded run, or sends one live request

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Replay the recorded run (no API calls, no key needed):

```bash
streamlit run app_streamlit.py
```

Run the experiment yourself (12 requests, about $0.40 at Standard rates). Put `OPENAI_API_KEY=...` in a `.env` file first and keep that file out of version control.

```bash
python checker.py        # verifies PNG and CSV hashes and prints the reference facts and pass rule
python describe.py       # writes new runs to recorded/live_outputs.json without overwriting recorded_outputs.json
python publish_page.py   # writes reviewed_page.html
```

The review fields and `reviewed_description` in `recorded/recorded_outputs.json` were added by hand after reading all 12 outputs. `python build_chart.py` rebuilds the PNG. `describe.py` refuses to run if either the PNG or CSV hash changes.
