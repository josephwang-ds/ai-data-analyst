# AI Data Analyst

A Streamlit app that turns any CSV into insights, charts, and recommendations using natural language — no SQL or code required.

## Overview

Upload a CSV, describe what you want to know, and get back data profiling, visualizations, and plain-English business recommendations — all in one step.

## Features

- **Auto profile** — scans schema, summarizes the dataset, flags data quality issues, and suggests the most relevant chart
- **Natural language Q&A** — ask any business question and get a direct answer, a chart, and a recommended next action
- **Built-in sample dataset** — weekly e-commerce sales by product, region, and channel; no upload needed to try it

## Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Data | Pandas |
| Visualization | Plotly |
| AI | DeepSeek `deepseek-chat` (structured JSON output) |

## Quickstart

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=your_deepseek_api_key
streamlit run app.py
```

API key can also be entered directly in the sidebar.

## ModelScope deployment

This app can be deployed to ModelScope Studio with the same codebase.

- SDK: `Streamlit`
- Entry file: `app.py`
- Dependency file: `requirements.txt`
- Runtime secret: `OPENAI_API_KEY` (or `DEEPSEEK_API_KEY`)

Detailed checklist: [`MODELSCOPE_DEPLOY.md`](./MODELSCOPE_DEPLOY.md)

## Example questions

- Which product has the worst ROI on ad spend?
- Is there a week-over-week trend in conversion rate?
- Which region is underperforming relative to its ad budget?

## Related

- [ChatBI](https://github.com/josephwang-ds/chatbi) — natural language → SQL → result table
- [A/B Test Analyzer](https://github.com/josephwang-ds/ab-test-analyzer) — experiment data → significance test → verdict

---

[josephjwang.com](https://josephjwang.com) · [github.com/josephwang-ds](https://github.com/josephwang-ds)
