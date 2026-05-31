"""
AI Data Analyst — Demo ①
Upload a CSV → ask questions in plain English → get analysis, charts, and business suggestions.

Business impact: Cuts ad-hoc analysis from days to minutes.
"""

import os
import io
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    provider = st.session_state.get("provider", "DeepSeek")
    api_key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Add your API key in the sidebar.")
        st.stop()
    if provider == "DeepSeek":
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    else:
        return OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def summarize_df(df: pd.DataFrame) -> str:
    """Compact schema + sample for the LLM context window."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    buf.write("Column types:\n")
    for col, dtype in df.dtypes.items():
        buf.write(f"  {col}: {dtype}\n")
    buf.write("\nFirst 5 rows (JSON):\n")
    buf.write(df.head(5).to_json(orient="records", date_format="iso"))
    buf.write("\n\nDescriptive stats (numeric):\n")
    buf.write(df.describe(include="number").to_string())
    return buf.getvalue()


def ask_llm(client: OpenAI, system: str, user: str, model: str | None = None) -> str:
    if model is None:
        model = "deepseek-chat" if st.session_state.get("provider","DeepSeek")=="DeepSeek" else "gemini-2.0-flash"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


def auto_profile(client: OpenAI, df: pd.DataFrame) -> dict:
    """Return JSON with: summary, key_metrics, anomalies, chart_suggestion."""
    schema = summarize_df(df)
    system = (
        "You are a senior business analyst. "
        "Given a dataset schema and sample, return ONLY valid JSON with these keys:\n"
        '  "summary": one-paragraph plain-English description of what the data is about,\n'
        '  "key_metrics": list of 3-5 most important columns/metrics to watch,\n'
        '  "anomalies": list of 2-3 potential data quality issues or outliers to flag,\n'
        '  "chart_suggestion": {"type": "bar|line|scatter|histogram", "x": "<col>", "y": "<col>", "color": "<col or null>", "title": "<title>"}\n'
        "Return JSON only — no markdown fences."
    )
    raw = ask_llm(client, system, schema)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # strip markdown fences if model added them anyway
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)


def answer_question(client: OpenAI, df: pd.DataFrame, question: str) -> dict:
    """
    Return JSON: {answer, chart_suggestion, business_recommendation}
    chart_suggestion may be null.
    """
    schema = summarize_df(df)
    system = (
        "You are a senior data analyst. The user will ask a business question about the dataset below. "
        "Provide a clear, specific answer with numbers where possible. "
        "Return ONLY valid JSON with:\n"
        '  "answer": detailed plain-English answer,\n'
        '  "chart_suggestion": {"type": "bar|line|scatter|histogram|box", "x": "<col>", "y": "<col>", "color": "<col or null>", "title": "<title>"} or null,\n'
        '  "business_recommendation": 2-3 sentence actionable recommendation.\n'
        "Return JSON only."
    )
    user = f"Dataset info:\n{schema}\n\nQuestion: {question}"
    raw = ask_llm(client, system, user)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)


def render_chart(df: pd.DataFrame, spec: dict | None) -> None:
    if not spec:
        return
    chart_type = spec.get("type", "bar")
    x, y = spec.get("x"), spec.get("y")
    color = spec.get("color") or None
    title = spec.get("title", "")

    # Validate columns exist
    valid_cols = set(df.columns)
    if x and x not in valid_cols:
        x = None
    if y and y not in valid_cols:
        y = None
    if color and color not in valid_cols:
        color = None

    try:
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x or y, color=color, title=title)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title)
        else:  # bar (default)
            # aggregate if needed
            if x and y and x in df.columns and y in df.columns:
                agg = df.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False).head(20)
                fig = px.bar(agg, x=x, y=y, color=color if color in agg.columns else None, title=title)
            else:
                return
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart could not render: {e}")


# ── Sample data ────────────────────────────────────────────────────────────────

SAMPLE_CSV = """date,product,category,revenue,units_sold,conversion_rate,ad_spend,region
2024-01-01,Widget A,Electronics,12500,250,0.032,3200,North
2024-01-01,Widget B,Electronics,8900,178,0.021,2800,South
2024-01-01,Gadget X,Accessories,4300,430,0.045,900,East
2024-01-01,Gadget Y,Accessories,3100,310,0.038,700,West
2024-01-08,Widget A,Electronics,13200,264,0.034,3400,North
2024-01-08,Widget B,Electronics,7600,152,0.019,2600,South
2024-01-08,Gadget X,Accessories,5100,510,0.052,950,East
2024-01-08,Gadget Y,Accessories,2900,290,0.036,680,West
2024-01-15,Widget A,Electronics,14100,282,0.037,3600,North
2024-01-15,Widget B,Electronics,9200,184,0.023,2900,South
2024-01-15,Gadget X,Accessories,4800,480,0.049,920,East
2024-01-15,Gadget Y,Accessories,3400,340,0.041,730,West
2024-01-22,Widget A,Electronics,11800,236,0.030,3100,North
2024-01-22,Widget B,Electronics,10100,202,0.025,3100,South
2024-01-22,Gadget X,Accessories,5500,550,0.056,980,East
2024-01-22,Gadget Y,Accessories,3800,380,0.044,760,West
2024-01-29,Widget A,Electronics,15200,304,0.039,3800,North
2024-01-29,Widget B,Electronics,8400,168,0.020,2700,South
2024-01-29,Gadget X,Accessories,6200,620,0.063,1020,East
2024-01-29,Gadget Y,Accessories,4100,410,0.048,800,West
"""

SAMPLE_QUESTIONS = [
    "Which product has the highest average revenue?",
    "Which region has the best conversion rate?",
    "Is there a correlation between ad spend and revenue?",
    "What is the week-over-week revenue trend for Widget A?",
    "Which product has the worst ROI on ad spend?",
]

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("📊 AI Data Analyst")
st.caption("Upload a CSV → ask questions in plain English → get analysis, charts, and business recommendations.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.radio("Model", ["DeepSeek", "Gemini Flash"], horizontal=True)
    st.session_state["provider"] = provider
    placeholder = "AIza..." if provider == "Gemini Flash" else "sk-..."
    label = "Google AI Studio Key" if provider == "Gemini Flash" else "DeepSeek API Key"
    api_key_input = st.text_input(label, type="password", placeholder=placeholder)
    if api_key_input:
        st.session_state["api_key"] = api_key_input

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload your CSV (or use the sample)\n"
        "2. The AI profiles your data automatically\n"
        "3. Ask any business question in plain English\n"
        "4. Get answers, charts, and next-step recommendations"
    )
    st.divider()
    st.markdown("**Business impact**")
    st.markdown("Cuts ad-hoc analysis turnaround from *days to minutes*, enabling self-serve data for non-technical stakeholders.")
    st.divider()
    st.markdown("Built by [Joseph Wang](https://josephjwang.com) · [GitHub](https://github.com/josephwang-ds/ai-data-analyst)")

# ── Data loading ──────────────────────────────────────────────────────────────
st.subheader("1 · Load your data")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
with col2:
    use_sample = st.button("▶ Use sample dataset", use_container_width=True)

df = None
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
elif use_sample or "sample_loaded" in st.session_state:
    st.session_state["sample_loaded"] = True
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))
    st.info("Using sample e-commerce dataset (weekly product sales, Jan 2024)")

if df is not None:
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # ── Auto profile ──────────────────────────────────────────────────────────
    st.subheader("2 · Auto analysis")

    profile_key = f"profile_{id(df)}"
    if profile_key not in st.session_state:
        if st.button("🔍 Run auto analysis", use_container_width=False):
            with st.spinner("Analyzing your dataset…"):
                client = get_client()
                try:
                    profile = auto_profile(client, df)
                    st.session_state[profile_key] = profile
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    if profile_key in st.session_state:
        profile = st.session_state[profile_key]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Dataset summary**")
            st.write(profile.get("summary", ""))

            st.markdown("**Key metrics to watch**")
            for m in profile.get("key_metrics", []):
                st.markdown(f"- {m}")

        with col_b:
            st.markdown("**Data quality flags**")
            for a in profile.get("anomalies", []):
                st.warning(a)

        st.markdown("**Suggested chart**")
        render_chart(df, profile.get("chart_suggestion"))

    # ── Q&A ───────────────────────────────────────────────────────────────────
    st.subheader("3 · Ask a business question")

    # Suggested questions
    st.markdown("**Quick picks:**")
    q_cols = st.columns(len(SAMPLE_QUESTIONS))
    chosen_q = None
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with q_cols[i]:
            if st.button(q, key=f"sq_{i}"):
                chosen_q = q

    question = st.text_input(
        "Or type your own question",
        value=chosen_q or "",
        placeholder="Which product has the highest conversion rate?",
    )

    if question and st.button("💬 Analyse", type="primary"):
        with st.spinner("Thinking…"):
            client = get_client()
            try:
                result = answer_question(client, df, question)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Failed: {e}")

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.divider()
        col_ans, col_rec = st.columns([3, 2])
        with col_ans:
            st.markdown("**Answer**")
            st.write(res.get("answer", ""))
        with col_rec:
            st.markdown("**Business recommendation**")
            st.info(res.get("business_recommendation", ""))

        render_chart(df, res.get("chart_suggestion"))

else:
    st.info("⬆ Upload a CSV or click **Use sample dataset** to get started.")
