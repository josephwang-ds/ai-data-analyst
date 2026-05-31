# AI Data Analyst — Demo ①

> **Business impact:** Cuts ad-hoc data analysis from days to minutes. Non-technical stakeholders self-serve answers instead of waiting for the data team.

A Streamlit app that turns any CSV into insights, charts, and business recommendations — powered by GPT-4o-mini.

## Live demo

[josephjwang.com/analyst](https://josephjwang.com/analyst)

## What it does

| Step | What happens |
|---|---|
| Upload CSV | Paste your data or use the built-in e-commerce sample |
| Auto profile | AI scans schema, flags data quality issues, picks the best chart |
| Ask anything | Plain-English question → answer + chart + actionable recommendation |

**Example questions:**
- "Which product has the worst ROI on ad spend?"
- "Is there a week-over-week trend in conversion rate?"
- "Which region is underperforming relative to its ad budget?"

## Tech stack

- **Frontend:** Streamlit
- **Analysis:** Pandas + Plotly
- **AI layer:** OpenAI GPT-4o-mini (structured JSON output)
- **No backend required** — runs locally or on Streamlit Cloud

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

Or enter your API key directly in the sidebar — no env var needed.

## Why this matters for hiring teams

This demo replicates a workflow I built internally at my e-commerce business:

- ~80% of routine data requests became self-serve
- Response time dropped from day-level to minute-level
- Non-technical ops team stopped waiting on the data analyst

The same pattern applies to any team where data requests bottleneck on one person.

## Related demos

- [ChatBI / Text-to-SQL](../chatbi) — natural language → SQL → result table
- [A/B Test Analyzer](../ab-test-analyzer) — upload experiment data → significance test → ship/no-ship recommendation

---

Built by [Joseph Wang](https://josephjwang.com) · Northwestern MSc Data Science · 6 years enterprise analytics
