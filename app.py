"""
AI Data Analyst — Demo ①
Upload a CSV → auto KPI dashboard → multi-turn Q&A → charts + recommendations.
"""

import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0f1117; }
[data-testid="stSidebar"] { background:#1a1f2e; border-right:1px solid #243042; }
[data-testid="stAppViewContainer"] .main .block-container { max-width:1220px; padding-top:1.2rem; }
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-size:16px; }
h1,h2,h3,p,label,div,span { color:#e2e8f0 !important; }
p, label, [data-testid="stMarkdownContainer"] p { font-size:0.95rem !important; }
.stButton>button {
    background:#1e293b;border:1px solid #475569;color:#e2e8f0 !important;border-radius:8px;
    min-height:42px;font-weight:600;
}
.stButton>button:hover { border-color:#818cf8;background:#2d3748; }
.stButton>button[kind="primary"] { background:#6366f1 !important;border-color:#6366f1 !important; }
[data-testid="stDownloadButton"]>button {
    background:#1e293b !important;border:1px solid #475569 !important;color:#e2e8f0 !important;
    border-radius:8px !important;min-height:42px;font-weight:600;
}
[data-testid="stDownloadButton"]>button:hover { border-color:#818cf8 !important; }
[data-testid="stFileUploader"] {
    border:2px dashed #475569 !important;border-radius:10px !important;background:#111827 !important;
    padding:1rem !important;
}
[data-testid="stFileUploaderDropzone"] { background:#111827 !important;border:0 !important; }
[data-testid="stFileUploaderDropzone"] * { color:#cbd5e1 !important; }
[data-testid="stDataFrame"] { border:1px solid #334155;border-radius:8px; }

/* Header gradient */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-label { color: #a5b4fc; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { color: #fff; font-size: 1.6rem; font-weight: 700; margin: 0.2rem 0; }
.kpi-delta { font-size: 0.78rem; }
.kpi-delta.pos { color: #34d399; }
.kpi-delta.neg { color: #f87171; }
.kpi-delta.neu { color: #94a3b8; }

/* Chat bubbles */
.chat-q {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #e2e8f0;
}
.chat-a {
    background: #0f172a;
    border-left: 3px solid #06b6d4;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0 1rem 0;
    font-size: 0.88rem;
    color: #94a3b8;
}

/* Section headers */
.section-tag {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

CHART_THEME = {
    "template": "plotly_dark",
    "color_sequence": ["#6366f1", "#06b6d4", "#34d399", "#f59e0b", "#f43f5e", "#a78bfa"],
    "paper_bgcolor": "#0f1117",
    "plot_bgcolor": "#0f1117",
}

# ── LLM helpers ────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ API key not configured. Please contact the demo owner.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def get_model() -> str:
    return "deepseek-chat"

def ask_llm(client: OpenAI, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        for prefix in ["```json", "```"]:
            cleaned = cleaned.removeprefix(prefix)
        cleaned = cleaned.removesuffix("```").strip()
        return json.loads(cleaned)

# ── Data intelligence ──────────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect column roles."""
    date_cols, numeric_cols, categorical_cols = [], [], []
    for col in df.columns:
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            numeric_cols.append(col)
        elif df[col].dtype == "object":
            # Try parsing as date
            try:
                pd.to_datetime(df[col].dropna().head(5))
                date_cols.append(col)
            except Exception:
                categorical_cols.append(col)
        else:
            try:
                date_cols.append(col)
            except Exception:
                pass
    return {"date": date_cols, "numeric": numeric_cols, "categorical": categorical_cols}

def compute_kpis(df: pd.DataFrame, numeric_cols: list) -> list[dict]:
    """Auto-compute KPI cards from numeric columns (top 4)."""
    kpis = []
    for col in numeric_cols[:4]:
        total = df[col].sum()
        mean = df[col].mean()
        kpis.append({"label": col.replace("_", " ").title(), "value": total, "mean": mean})
    return kpis

def summarize_df(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    buf.write("Columns and types:\n")
    for col, dtype in df.dtypes.items():
        sample_vals = df[col].dropna().head(3).tolist()
        buf.write(f"  {col} ({dtype}): sample = {sample_vals}\n")
    buf.write("\nDescriptive stats (numeric):\n")
    buf.write(df.describe(include="number").round(2).to_string())
    if len(df) > 5:
        buf.write("\n\nFirst 5 rows:\n")
        buf.write(df.head(5).to_json(orient="records", date_format="iso"))
    return buf.getvalue()

# ── LLM tasks ──────────────────────────────────────────────────────────────────

def auto_profile(client: OpenAI, df: pd.DataFrame) -> dict:
    schema = summarize_df(df)
    system = (
        "You are a senior business analyst. Analyze this dataset and return ONLY valid JSON:\n"
        '{\n'
        '  "summary": "one paragraph describing what this data is about and key patterns",\n'
        '  "key_metrics": ["metric1", "metric2", "metric3"],\n'
        '  "anomalies": ["observation1", "observation2"],\n'
        '  "insights": ["insight1", "insight2", "insight3"],\n'
        '  "chart_suggestions": [\n'
        '    {"type": "bar|line|scatter|histogram", "x": "<col>", "y": "<col>", "color": "<col or null>", "title": "<title>"},\n'
        '    {"type": "bar|line|scatter|histogram", "x": "<col>", "y": "<col>", "color": "<col or null>", "title": "<title>"}\n'
        '  ]\n'
        '}\n'
        "Return JSON only — no markdown fences."
    )
    raw = ask_llm(client, system, [{"role": "user", "content": schema}], max_tokens=1500)
    return parse_json(raw)

def answer_question(client: OpenAI, df: pd.DataFrame, question: str) -> dict:
    schema = summarize_df(df)
    history = st.session_state.get("chat_history", [])

    # Build conversation context
    messages = []
    for turn in history[-4:]:  # last 4 turns for context
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": f"Dataset:\n{schema}\n\nQuestion: {question}"})

    system = (
        "You are a senior data analyst with access to the dataset described. "
        "Answer business questions with specific numbers. "
        "You have memory of the conversation history — use it for follow-up questions. "
        "Return ONLY valid JSON:\n"
        '{\n'
        '  "answer": "detailed answer with specific numbers",\n'
        '  "chart_suggestion": {"type": "bar|line|scatter|histogram|box", "x": "<col>", "y": "<col>", "color": "<col or null>", "title": "<title>"} or null,\n'
        '  "business_recommendation": "2-3 sentence actionable recommendation"\n'
        '}\n'
        "Return JSON only."
    )
    raw = ask_llm(client, system, messages, max_tokens=1400)
    return parse_json(raw)

# ── Chart rendering ────────────────────────────────────────────────────────────

def render_chart(df: pd.DataFrame, spec: dict | None, height: int = 380) -> None:
    if not spec:
        return
    chart_type = spec.get("type", "bar")
    x, y = spec.get("x"), spec.get("y")
    color = spec.get("color") or None
    title = spec.get("title", "")
    valid_cols = set(df.columns)
    if x and x not in valid_cols: x = None
    if y and y not in valid_cols: y = None
    if color and color not in valid_cols: color = None

    kwargs = dict(
        color_discrete_sequence=CHART_THEME["color_sequence"],
        template=CHART_THEME["template"],
        title=title,
    )
    try:
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, markers=True, **kwargs)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, **kwargs)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x or y, color=color, **kwargs)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, **kwargs)
        else:
            if x and y and x in df.columns and y in df.columns:
                agg = df.groupby(x)[y].sum().reset_index().sort_values(y, ascending=False).head(15)
                fig = px.bar(agg, x=x, y=y, color=color if color in agg.columns else None, **kwargs)
            else:
                return
        fig.update_layout(
            height=height,
            paper_bgcolor=CHART_THEME["paper_bgcolor"],
            plot_bgcolor=CHART_THEME["plot_bgcolor"],
            font=dict(color="#e2e8f0"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart error: {e}")

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
    "Which product has the highest ROI on ad spend?",
    "What is the revenue trend week over week?",
    "Which region is underperforming?",
    "Compare conversion rates across products",
    "Where should we increase ad budget?",
]

# ── UI ─────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown('<p class="hero-title">📊 AI Data Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload any CSV → instant KPI dashboard → multi-turn AI analysis → actionable recommendations</p>', unsafe_allow_html=True)
st.markdown(
    """
<div style="background:#1a1f2e;border:1px solid #334155;border-radius:10px;padding:0.9rem 1.1rem;color:#cbd5e1;line-height:1.75;margin-bottom:0.8rem;font-size:0.86rem;">
<b>Demo storyline</b><br>
1) Start with KPI snapshot to frame business health<br>
2) Use one trend chart to explain what changed and where<br>
3) End with one AI-backed action recommendation (budget, region, or product focus)<br><br>
<b>Suggested flow (2-4 min)</b>: sample data start → ask 2 follow-ups → upload one CSV for adaptation proof
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 AI Data Analyst")
    st.caption("Powered by DeepSeek · No sign-up required")

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload any CSV — or try the sample\n"
        "2. Auto KPI dashboard renders instantly\n"
        "3. Click **Run AI Analysis** for insights\n"
        "4. Ask follow-up questions in plain English"
    )

    st.divider()
    st.markdown("**📂 CSV format**")
    st.caption("Any CSV works — the AI auto-detects columns. Download the sample to see an example.")
    st.download_button(
        "⬇ Download sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_ecommerce.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    # Chat history
    if st.session_state.get("chat_history"):
        st.markdown("### 💬 History")
        for i, turn in enumerate(st.session_state["chat_history"]):
            with st.expander(f"Q{i+1}: {turn['question'][:35]}…", expanded=False):
                st.caption(turn["answer"][:180] + "…" if len(turn["answer"]) > 180 else turn["answer"])
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
        st.divider()

    if st.button("♻️ Reset demo state", use_container_width=True):
        for key in [
            "sample_loaded",
            "_q_inject",
            "question_input",
            "chat_history",
            "last_result",
            "session_summary",
        ]:
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()

    st.markdown("**Stack:** Streamlit · Pandas · Plotly · DeepSeek")
    st.markdown("**[GitHub](https://github.com/josephwang-ds/ai-data-analyst)** · **[josephjwang.com](https://josephjwang.com)**")

# ── 1. Data loading ────────────────────────────────────────────────────────────
st.markdown('<span class="section-tag">Step 1 — Load Data</span>', unsafe_allow_html=True)
st.caption("Upload any CSV — the AI auto-detects column types. Need a starting point? Download the sample CSV from the sidebar.")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
with col2:
    use_sample = st.button("▶ Use sample data", use_container_width=True, type="secondary")

df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV. Please check encoding/delimiter. Detail: {e}")
        df = None
    if df is not None:
        if df.empty:
            st.error("Uploaded CSV is empty. Please upload a file with data rows.")
            df = None
        else:
            st.success(f"✅ Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
            if "sample_loaded" in st.session_state:
                del st.session_state["sample_loaded"]
elif use_sample or "sample_loaded" in st.session_state:
    st.session_state["sample_loaded"] = True
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))
    st.info("📦 Sample dataset: weekly e-commerce sales (4 products × 5 weeks, Jan 2024)")

if df is not None:
    cols = detect_columns(df)
    st.caption(
        f"Detected columns — date: {len(cols['date'])}, numeric: {len(cols['numeric'])}, categorical: {len(cols['categorical'])}"
    )

    with st.expander("🔍 Data preview", expanded=False):
        tab1, tab2 = st.tabs(["Table", "Stats"])
        with tab1:
            st.dataframe(df, use_container_width=True, height=220)
        with tab2:
            st.dataframe(df.describe(include="all").round(2), use_container_width=True)

    # ── 2. KPI Dashboard ──────────────────────────────────────────────────────
    st.markdown('<span class="section-tag">Step 2 — KPI Dashboard</span>', unsafe_allow_html=True)

    kpis = compute_kpis(df, cols["numeric"])
    if kpis:
        kpi_cols = st.columns(len(kpis))
        for i, kpi in enumerate(kpis):
            with kpi_cols[i]:
                val = kpi["value"]
                formatted = f"${val:,.0f}" if "revenue" in kpi["label"].lower() or "spend" in kpi["label"].lower() else f"{val:,.1f}"
                st.metric(
                    label=kpi["label"],
                    value=formatted,
                    delta=f"avg {kpi['mean']:,.1f}",
                )

    # Auto trend chart if date column detected
    if cols["date"] and cols["numeric"]:
        date_col = cols["date"][0]
        top_numeric = cols["numeric"][0]
        try:
            trend_df = df.copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col])
            agg_col = cols["categorical"][0] if cols["categorical"] else None
            if agg_col:
                trend_agg = trend_df.groupby([date_col, agg_col])[top_numeric].sum().reset_index()
                fig = px.line(
                    trend_agg, x=date_col, y=top_numeric, color=agg_col,
                    title=f"{top_numeric.replace('_',' ').title()} Trend",
                    color_discrete_sequence=CHART_THEME["color_sequence"],
                    template=CHART_THEME["template"],
                    markers=True,
                )
            else:
                trend_agg = trend_df.groupby(date_col)[top_numeric].sum().reset_index()
                fig = px.line(
                    trend_agg, x=date_col, y=top_numeric,
                    title=f"{top_numeric.replace('_',' ').title()} Trend",
                    color_discrete_sequence=CHART_THEME["color_sequence"],
                    template=CHART_THEME["template"],
                    markers=True,
                )
            fig.update_layout(
                height=320,
                paper_bgcolor=CHART_THEME["paper_bgcolor"],
                plot_bgcolor=CHART_THEME["plot_bgcolor"],
                font=dict(color="#e2e8f0"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # ── 3. AI Auto Profile ────────────────────────────────────────────────────
    st.markdown('<span class="section-tag">Step 3 — AI Profile</span>', unsafe_allow_html=True)

    profile_key = f"profile_{df.shape}"
    if profile_key not in st.session_state:
        if st.button("🤖 Run AI Analysis", type="primary", use_container_width=False):
            with st.spinner("Analyzing dataset with AI…"):
                client = get_client()
                try:
                    profile = auto_profile(client, df)
                    st.session_state[profile_key] = profile
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    if profile_key in st.session_state:
        profile = st.session_state[profile_key]

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("**📋 Summary**")
            st.write(profile.get("summary", ""))

            if profile.get("insights"):
                st.markdown("**💡 Key Insights**")
                for insight in profile.get("insights", []):
                    st.markdown(f"→ {insight}")

        with col_b:
            if profile.get("key_metrics"):
                st.markdown("**📌 Metrics to watch**")
                for m in profile.get("key_metrics", []):
                    st.markdown(f"• {m}")

            if profile.get("anomalies"):
                st.markdown("**⚠️ Flags**")
                for a in profile.get("anomalies", []):
                    st.warning(a)

        # Multiple chart suggestions
        suggestions = profile.get("chart_suggestions", [])
        if suggestions:
            tabs = st.tabs([f"Chart {i+1}" for i in range(len(suggestions))])
            for i, spec in enumerate(suggestions):
                with tabs[i]:
                    render_chart(df, spec)

    # ── 4. Multi-turn Q&A ─────────────────────────────────────────────────────
    st.markdown('<span class="section-tag">Step 4 — Ask Anything</span>', unsafe_allow_html=True)

    # Quick picks — use staging key to avoid post-widget injection error
    st.markdown("**Quick picks:**")
    q_cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with q_cols[i]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["_q_inject"] = q
                st.session_state.pop("last_result", None)
                st.rerun()

    # Transfer staging key before widget instantiation
    if "_q_inject" in st.session_state:
        st.session_state["question_input"] = st.session_state.pop("_q_inject")

    question = st.text_input(
        "Ask a business question",
        placeholder="e.g. Which product should we double down on next quarter?",
        key="question_input",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 6])
    with col_btn1:
        analyse = st.button("💬 Ask", type="primary", disabled=not question, use_container_width=True)
    with col_btn2:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state.pop("last_result", None)
            st.rerun()

    if analyse and question:
        with st.spinner("Thinking…"):
            client = get_client()
            try:
                result = answer_question(client, df, question)
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                st.session_state["chat_history"].append({
                    "question": question,
                    "answer": result.get("answer", ""),
                    "chart_spec": result.get("chart_suggestion"),
                    "recommendation": result.get("business_recommendation", ""),
                    "follow_ups": result.get("follow_up_questions", []),
                })
                st.session_state["last_result"] = result
                st.session_state["_q_inject"] = ""
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    # Latest result display
    if st.session_state.get("last_result"):
        res = st.session_state["last_result"]
        st.divider()
        st.markdown(
            "<div style='background:#111827;border:1px solid #334155;border-radius:10px;padding:0.7rem 1rem;color:#cbd5e1;font-size:0.84rem;margin-bottom:0.9rem;'>"
            "<b>Decision flow</b>: conclusion (what happened) → evidence (numbers/chart) → action (what to do next week)."
            "</div>",
            unsafe_allow_html=True,
        )

        # Answer card
        st.markdown("**💬 Answer**")
        st.markdown(
            f"<div style='background:#1e293b;border-left:3px solid #6366f1;border-radius:0 8px 8px 0;"
            f"padding:1rem 1.2rem;color:#e2e8f0;line-height:1.7;margin-bottom:0.8rem'>"
            f"{res.get('answer','')}</div>",
            unsafe_allow_html=True,
        )

        # Recommendation card
        rec = res.get("business_recommendation", "")
        if rec:
            st.markdown(
                f"<div style='background:#0c2a1f;border-left:3px solid #34d399;border-radius:0 8px 8px 0;"
                f"padding:0.8rem 1.2rem;color:#86efac;font-size:0.9rem;margin-bottom:1rem'>"
                f"🎯 <b>Recommendation:</b> {rec}</div>",
                unsafe_allow_html=True,
            )

        render_chart(df, res.get("chart_suggestion"))

    # ── Session Summary ────────────────────────────────────────────────────────
    history = st.session_state.get("chat_history", [])
    if len(history) >= 1:
        st.divider()
        col_sum, col_exp = st.columns([1, 1])
        with col_sum:
            if st.button("📋 Generate Session Summary", use_container_width=True):
                with st.spinner("Synthesizing analysis…"):
                    client = get_client()
                    qa_log = "\n\n".join(
                        f"Q: {t['question']}\nA: {t['answer']}"
                        + (f"\nRecommendation: {t['recommendation']}" if t.get("recommendation") else "")
                        for t in history
                    )
                    system = (
                        "You are a senior business analyst. Given a Q&A session over a dataset, "
                        "write a concise executive summary (3-5 paragraphs) covering: "
                        "key findings, patterns identified, risks or underperformers flagged, "
                        "and 3 prioritized action items. Be specific with numbers. Professional tone."
                    )
                    messages = [{"role": "user", "content": f"Dataset schema:\n{summarize_df(df)}\n\nQ&A Session:\n{qa_log}\n\nWrite the executive summary."}]
                    summary = ask_llm(client, system, messages, max_tokens=800)
                    st.session_state["session_summary"] = summary

        with col_exp:
            if st.session_state.get("session_summary") and history:
                lines = [
                    "# Data Analysis — Executive Summary",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
                    "## Executive Summary\n",
                    st.session_state["session_summary"],
                    "\n---\n## Full Q&A Log\n",
                ]
                for idx, t in enumerate(history, 1):
                    lines.append(f"### Q{idx}: {t['question']}")
                    lines.append(f"{t['answer']}\n")
                    if t.get("recommendation"):
                        lines.append(f"**Recommendation:** {t['recommendation']}\n")
                st.download_button(
                    "⬇ Export Report",
                    "\n".join(lines),
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        if st.session_state.get("session_summary"):
            st.markdown(
                f"<div style='background:#1a1f2e;border:1px solid #334155;border-radius:10px;"
                f"padding:1.2rem 1.5rem;color:#e2e8f0;line-height:1.8;margin-top:1rem'>"
                f"<div style='font-size:0.75rem;font-weight:600;letter-spacing:0.08em;color:#94a3b8;"
                f"text-transform:uppercase;margin-bottom:0.8rem'>Executive Summary</div>"
                f"{st.session_state['session_summary'].replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #64748b;">
        <div style="font-size:3rem">📂</div>
        <p style="font-size:1.1rem; margin-top:0.5rem">Upload a CSV or click <b>Use sample data</b> to get started</p>
    </div>
    """, unsafe_allow_html=True)
