"""
app.py
======
The Legal Narrative Bias Explorer — final user-friendly version.

5 tabs:
    1. Search the Opinion
    2. How Faithful is the Coverage?
    3. How Did the Framing Change?
    4. Compare News Articles
    5. Whose Voice Does the Coverage Echo?

This version removes the optional-feature toggles (stance detection,
LLM-generated answers, LLM-as-judge) from the sidebar to keep the UI
simple. The underlying code in src/ still supports them — they just
default to off.
"""
from pathlib import Path
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src import config
from src.utils import file_hash
from src.rag import Retriever
from src.rouge_metrics import analyze_news_article
from src.semantic import load_glove, analyze_tone, radar_chart
from src.emotionality import emotionality_score
from src.topic_model import fit_topics, document_topic_distribution
from src.relevance import check_relevance
from src.article_store import (
    list_articles, save_article, SAMPLE_DIR, LEAN_COLORS,
)
from src.comparison import analyze_one_article
from src.justice_voice import detect_sections, voice_summary, section_texts_by_voice


# ============================================================================
# Friendly label maps
# ============================================================================
AXIS_FRIENDLY = {
    "Reason ↔ Emotion": "Analytical ↔ Emotional",
    "Abstract ↔ Concrete": "General concepts ↔ Specific people/places",
    "Passive ↔ Active": "Cautious wording ↔ Strong wording",
}

# Lean labels — long-form descriptions per user request
LEAN_LABELS = {
    "left": "Left-leaning — Editorial board generally aligns with progressive/Democratic positions",
    "center": "Center / wire service — Aims for neutral reporting; less editorial slant",
    "right": "Right-leaning — Editorial board generally aligns with conservative/Republican positions",
}
LEAN_VALUES = list(LEAN_LABELS.keys())
LEAN_VALUES_BY_LABEL = {v: k for k, v in LEAN_LABELS.items()}

# Optional features are hardcoded OFF (toggles removed from sidebar per user request)
ENABLE_STANCE = False
USE_LLM = False
ENABLE_JUDGE = False


def friendly_axis(name: str) -> str:
    return AXIS_FRIENDLY.get(name, name)


# ============================================================================
# Page setup
# ============================================================================
st.set_page_config(
    page_title="Legal Narrative Bias Explorer",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ The Legal Narrative Bias Explorer")
st.caption(
    "Compare media coverage to the actual judicial opinion across multiple "
    "dimensions: factual content, factual deviation, tone, topic emphasis, "
    "and judicial voice alignment."
)


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("Court Opinion")

    uploaded = st.file_uploader("Upload a court opinion PDF", type=["pdf"])
    existing_pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    selected_existing = None
    if existing_pdfs:
        st.markdown("**Or pick a previously uploaded case:**")
        names = [p.name for p in existing_pdfs]
        selected_existing = st.selectbox(
            "Existing cases", options=["—"] + names, label_visibility="collapsed",
        )

    pdf_path = None
    if uploaded is not None:
        pdf_path = config.RAW_DIR / uploaded.name
        pdf_path.write_bytes(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name}")
    elif selected_existing and selected_existing != "—":
        pdf_path = config.RAW_DIR / selected_existing

    st.divider()
    st.header("Settings")
    top_k = st.slider("Number of matching passages to show", 3, 10, config.DEFAULT_TOP_K)
    oracle_max = st.slider("Maximum length of best-faithful summary", 5, 30, 15)

    st.markdown("**Word vector file path**")
    glove_path_str = st.text_input(
        "GloVe file path", value=str(config.GLOVE_PATH), label_visibility="collapsed",
    )


if pdf_path is None or not pdf_path.exists():
    st.info("👈 Upload a court opinion PDF in the sidebar to begin.")
    st.stop()


# ============================================================================
# Cached resource loaders
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_retriever(pdf_path_str: str, _signature: str) -> Retriever:
    return Retriever(Path(pdf_path_str))


@st.cache_resource(show_spinner=False)
def get_glove(path_str: str):
    return load_glove(Path(path_str))


@st.cache_resource(show_spinner=False)
def get_topic_model(_signature: str, num_topics: int = 8):
    retriever_local = get_retriever(str(pdf_path), _signature)
    raw = retriever_local.vectorstore._collection.get(include=["documents"])
    return fit_topics(raw["documents"], num_topics=num_topics)


with st.spinner(f"Loading vector store for {pdf_path.name}…"):
    sig = file_hash(pdf_path)
    retriever = get_retriever(str(pdf_path), sig)

st.success(f"📚 Loaded **{pdf_path.name}** — ready to analyze.")


# ============================================================================
# Tabs
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Search the Opinion",
    "How Faithful is the Coverage?",
    "How Did the Framing Change?",
    "Compare News Articles",
    "Whose Voice Does the Coverage Echo?",
])


# ----------------------------------------------------------------------------
# TAB 1: Search the Opinion
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Ask the opinion a question")
    st.caption(
        "Type a question and we'll show you the parts of the ruling that best "
        "match your question, with page numbers."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g., What is the standard of review for abortion regulations?",
    )
    if question:
        with st.spinner("Searching the opinion…"):
            passages = retriever.query(question, top_k=top_k)
        st.markdown(f"### Top {len(passages)} matches from the opinion")
        for i, p in enumerate(passages, 1):
            with st.expander(
                f"**Match {i}** — page {p['page']} | relevance: {p['similarity']:.2f}",
                expanded=(i <= 2),
            ):
                st.markdown(p["text"])


# ----------------------------------------------------------------------------
# TAB 2: How Faithful is the Coverage?
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("How factually faithful is the news coverage?")
    st.caption(
        "Paste a news article. We'll find the sentences from the ruling that best "
        "match what the article is saying, and measure how closely the article "
        "wording matches those source sentences."
    )

    news_text = st.text_area(
        "News article text",
        height=250,
        key="rouge_article_text",
        placeholder="Paste the full text of a news article here.",
    )

    if st.button("Run analysis", type="primary", key="rouge_run"):
        if not news_text.strip():
            st.warning("Paste a news article first.")
        else:
            with st.spinner("Loading the ruling…"):
                opinion_text = retriever.get_full_opinion_text()

            with st.spinner("Checking that the article is about this ruling…"):
                relevance = check_relevance(
                    news_text=news_text,
                    vectorstore=retriever.vectorstore,
                    opinion_full_text=opinion_text,
                )

            with st.spinner("Running analysis…"):
                t0 = time.time()
                result = analyze_news_article(
                    opinion_text=opinion_text,
                    news_text=news_text,
                    max_oracle_sentences=oracle_max,
                )
                elapsed = time.time() - t0

            st.session_state["rouge_result"] = result
            st.session_state["rouge_relevance"] = relevance
            st.session_state["rouge_news_text"] = news_text
            st.session_state["rouge_elapsed"] = elapsed

    # Render persisted results
    if "rouge_result" in st.session_state:
        result = st.session_state["rouge_result"]
        relevance = st.session_state["rouge_relevance"]
        elapsed = st.session_state.get("rouge_elapsed", 0)

        if not relevance["is_relevant"]:
            st.error(f"**{relevance['verdict']}**")
            st.warning(relevance["suggestion"])
            with st.expander("Why this was flagged"):
                st.write(f"Match to ruling: **{relevance['similarity']}**")
                st.write(f"Key terms found: **{relevance['term_overlap']*100:.0f}%**")
                if relevance["key_terms"]:
                    st.write(f"Key terms looked for: {', '.join(relevance['key_terms'][:5])}…")
        else:
            st.success(f"{relevance['verdict']} (match score: {relevance['similarity']})")

        # Headline metrics
        col_a, col_b = st.columns(2)
        col_a.metric(
            "Sentences pulled from the ruling",
            result["num_oracle_sentences"],
            help="The few sentences in the ruling that, taken together, are the closest possible match to your news article."
        )
        col_b.metric(
            "Faithfulness score (0 – 1)",
            f"{result['rouge_news_vs_oracle']['rougeL']['f1']:.3f}",
            help="How closely the news article's wording matches the best-matching sentences from the ruling. Higher = more faithful."
        )

        # Show statistics — collapsed by default
        with st.expander("📊 Show statistics — detailed comparison breakdown"):
            st.markdown(
                """
**What this table shows**

We run three different comparisons, each measured three different ways. Here's what each row means.

**The three comparisons (rows):**

1. **News article vs. full ruling.** We compare your news article against the entire ruling. *"How much of the news appears in the source"* will usually be very high. Most words in the article appear somewhere in a long legal document. *"How much of the source appears in the news"* will be very low, because a short news article physically cannot contain a full ruling that is many times longer. This comparison alone is unfair due to length mismatch, which is why we have the next one.

2. **News article vs. matching ruling sentences (the headline comparison).** We first find the few sentences from the ruling that best match the article, shown below the table as "best-matching sentences." Then we compare the article to just those sentences. Now both texts are similar in length, so this is the fair comparison. The "Overall match" on the Longest matched sequence row is your Faithfulness score above.

3. **Matching sentences vs. full ruling (sanity check).** This just verifies that the matching sentences really came from the ruling. The numbers are supposed to be small, because the matching sentences are a tiny subset of the full ruling. If this number were high, it would mean we accidentally picked a large portion of the ruling, which would be wrong.

**The three measurement types (per row):**

* **Single words.** Counts unique words shared between the two texts. Loosest check.
* **Word pairs.** Counts consecutive word pairs shared. Stricter, because it punishes paraphrasing.
* **Longest matched sequence.** The longest run of consecutive words shared in order. Strictest check, rewards direct quotation.

A typical paraphrase pattern (what most centrist news coverage looks like) is when the single words score is moderate (same vocabulary as the ruling), but the longest matched sequence is much lower (different sentence structure). That's the journalist using the ruling's vocabulary while restructuring the sentences in their own voice.
                """
            )

            metric_friendly = {
                "rouge1": "Single words",
                "rouge2": "Word pairs",
                "rougeL": "Longest matched sequence",
            }
            comparison_friendly = {
                "rouge_news_vs_opinion": "News article vs. full ruling",
                "rouge_news_vs_oracle": "News article vs. matching ruling sentences",
                "rouge_oracle_vs_opinion": "Matching sentences vs. full ruling (sanity check)",
            }

            rows = []
            for key, comp_label in comparison_friendly.items():
                for metric in ["rouge1", "rouge2", "rougeL"]:
                    s = result[key][metric]
                    rows.append({
                        "What's being compared": comp_label,
                        "Type of word-match check": metric_friendly[metric],
                        "How much of the news appears in the source": f"{s['precision']:.3f}",
                        "How much of the source appears in the news": f"{s['recall']:.3f}",
                        "Overall match": f"{s['f1']:.3f}",
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Best-matching sentences from the ruling")
        st.caption("These are the sentences in the ruling whose wording most closely matches the news article.")
        st.info(result["oracle_summary"])

        # Save section
        st.divider()
        with st.expander("💾 Save this article for news comparison"):
            col1, col2 = st.columns(2)
            with col1:
                save_outlet = st.text_input("Outlet name", value="", key="save_outlet_t2")
                save_lean_label = st.selectbox(
                    "Political lean",
                    options=list(LEAN_LABELS.values()),
                    key="save_lean_t2",
                )
                save_lean = LEAN_VALUES_BY_LABEL[save_lean_label]
            with col2:
                save_title = st.text_input("Headline (optional)", key="save_title_t2")
                save_url = st.text_input("URL (optional)", key="save_url_t2")
            if st.button("Save", key="save_btn_t2"):
                if save_outlet.strip():
                    p = save_article(
                        body=st.session_state["rouge_news_text"],
                        outlet=save_outlet, lean=save_lean,
                        title=save_title, url=save_url,
                    )
                    st.success(f"Saved to `sample_articles/{p.name}`")
                else:
                    st.warning("Please enter an outlet name.")


# ----------------------------------------------------------------------------
# TAB 3: How Did the Framing Change?
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("How did the framing change?")
    st.caption(
        "Two ways to measure how the news *sounds* compared to the ruling itself: "
        "tone profile across dimensions, and a single emotional intensity score."
    )

    with st.expander("How are the tone dimensions measured?"):
        st.markdown(
            "We use three dimensions, each defined by anchor words at both ends. "
            "We measure where each document falls between the two ends:"
        )
        for axis_name, spec in config.SEMANTIC_AXES.items():
            friendly = friendly_axis(axis_name)
            st.markdown(f"**{friendly}**")
            st.markdown(
                f"- Toward the left side: `{', '.join(spec['negative_pole'])}`  \n"
                f"- Toward the right side: `{', '.join(spec['positive_pole'])}`"
            )

    news_for_tone = st.text_area(
        "News article text",
        height=200,
        key="tone_article_text",
        placeholder="Paste the full text of a news article here.",
    )

    chart_style = st.radio(
        "Chart style",
        options=["Bar chart", "Radar chart"],
        horizontal=True,
    )

    if st.button("Run tone analysis", type="primary", key="tone_run"):
        glove_path = Path(glove_path_str)
        if not glove_path.exists():
            st.error(f"Word vector file not found at {glove_path}.")
        elif not news_for_tone.strip():
            st.warning("Paste a news article first.")
        else:
            with st.spinner("Loading…"):
                glove = get_glove(str(glove_path))

            opinion_for_tone = retriever.get_full_opinion_text()

            with st.spinner("Checking that the article is about this ruling…"):
                relevance = check_relevance(
                    news_text=news_for_tone,
                    vectorstore=retriever.vectorstore,
                    opinion_full_text=opinion_for_tone,
                )

            with st.spinner("Running analysis…"):
                scores = analyze_tone(
                    {"Court opinion": opinion_for_tone, "News article": news_for_tone},
                    glove,
                )

            with st.spinner("Computing emotional intensity…"):
                emo_news = emotionality_score(news_for_tone, glove)
                emo_opinion = emotionality_score(opinion_for_tone, glove)

            st.session_state["tone_scores"] = scores
            st.session_state["tone_emo_news"] = emo_news
            st.session_state["tone_emo_opinion"] = emo_opinion
            st.session_state["tone_relevance"] = relevance
            st.session_state["tone_glove_size"] = len(glove)
            st.session_state["tone_news_text"] = news_for_tone

    if "tone_scores" in st.session_state:
        scores = st.session_state["tone_scores"]
        emo_news = st.session_state["tone_emo_news"]
        emo_opinion = st.session_state["tone_emo_opinion"]
        relevance = st.session_state["tone_relevance"]

        if not relevance["is_relevant"]:
            st.warning(f"{relevance['verdict']} — {relevance['suggestion']}")
        else:
            st.success(relevance["verdict"])

        # Tone profile table
        st.markdown("### Tone profile: news vs. ruling")
        st.caption(
            "Each row is a tone dimension. Negative numbers lean toward the left "
            "side of the dimension; positive numbers lean toward the right."
        )
        df_rows = []
        for axis_name, doc_scores in scores.items():
            df_rows.append({
                "Tone dimension": friendly_axis(axis_name),
                "Court ruling": round(doc_scores["Court opinion"], 4),
                "News article": round(doc_scores["News article"], 4),
                "Shift (news vs. ruling)": round(
                    doc_scores["News article"] - doc_scores["Court opinion"], 4
                ),
            })
        st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

        # Emotional intensity
        st.markdown("### Emotional intensity score")
        st.caption(
            "A single number measuring how emotional vs. analytical each text is. "
            "Negative values mean more analytical; positive values mean more emotional."
        )
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Court ruling", f"{emo_opinion['emotionality']:+.4f}")
        col_e2.metric("News article", f"{emo_news['emotionality']:+.4f}")
        delta = emo_news["emotionality"] - emo_opinion["emotionality"]
        col_e3.metric("Shift (news vs. ruling)", f"{delta:+.4f}", delta=f"{delta:+.4f}")

        # Chart — bar OR radar
        if chart_style == "Bar chart":
            st.markdown("### Tone comparison (bar chart)")
            fig_bar = go.Figure()
            axis_names_friendly = [friendly_axis(a) for a in scores.keys()]
            opinion_vals = [scores[a]["Court opinion"] for a in scores.keys()]
            news_vals = [scores[a]["News article"] for a in scores.keys()]
            fig_bar.add_trace(go.Bar(
                name="Court ruling", y=axis_names_friendly, x=opinion_vals,
                orientation="h", marker_color="#6B7280",
            ))
            fig_bar.add_trace(go.Bar(
                name="News article", y=axis_names_friendly, x=news_vals,
                orientation="h", marker_color="#3B82F6",
            ))
            fig_bar.update_layout(
                barmode="group",
                xaxis_title="Score (negative = left side of dimension, positive = right side)",
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown("### Tone profile (radar chart)")
            friendly_scores = {friendly_axis(k): v for k, v in scores.items()}
            fig = radar_chart(friendly_scores, title="News vs. Court Ruling: Tone Profile")
            st.plotly_chart(fig, use_container_width=True)

        # Save section
        st.divider()
        with st.expander("💾 Save this article for news comparison"):
            col1, col2 = st.columns(2)
            with col1:
                save_outlet_t3 = st.text_input("Outlet name", value="", key="save_outlet_t3")
                save_lean_label_t3 = st.selectbox(
                    "Political lean",
                    options=list(LEAN_LABELS.values()),
                    key="save_lean_t3",
                )
                save_lean_t3 = LEAN_VALUES_BY_LABEL[save_lean_label_t3]
            with col2:
                save_title_t3 = st.text_input("Headline (optional)", key="save_title_t3")
                save_url_t3 = st.text_input("URL (optional)", key="save_url_t3")
            if st.button("Save", key="save_btn_t3"):
                if save_outlet_t3.strip():
                    p = save_article(
                        body=st.session_state["tone_news_text"],
                        outlet=save_outlet_t3, lean=save_lean_t3,
                        title=save_title_t3, url=save_url_t3,
                    )
                    st.success(f"Saved to `sample_articles/{p.name}`")
                else:
                    st.warning("Please enter an outlet name.")


# ----------------------------------------------------------------------------
# TAB 4: Compare News Articles
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Compare news articles")
    st.caption(
        "Articles you saved from Tab 2 or Tab 3 appear here. Click 'Run analysis' "
        "to score every article against the ruling and see how outlets differ."
    )

    articles = list_articles()
    st.markdown(f"**Found {len(articles)} saved article(s):**")

    if not articles:
        st.info(
            "No articles saved yet. Use the 'Save this article for news comparison' "
            "section in Tabs 2 or 3 to add articles here."
        )
    else:
        summary_df = pd.DataFrame([
            {
                "File": a.filename,
                "Outlet": a.outlet,
                "Political lean": LEAN_LABELS.get(a.lean, a.lean),
                "Words": a.word_count,
                "Title": a.title[:50],
            }
            for a in articles
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Run + Clear buttons side by side
        run_col, clear_col, _spacer = st.columns([1, 1, 4])
        with run_col:
            run_clicked = st.button("Run analysis", type="primary", key="t4_run_btn")
        with clear_col:
            clear_clicked = st.button("Clear previous results", key="t4_clear_btn")

        if clear_clicked:
            st.session_state["t4_confirm_clear"] = True

        if st.session_state.get("t4_confirm_clear"):
            st.warning(
                "**This will permanently delete:**  \n"
                "• All cached comparison results in this session  \n"
                f"• All {len(articles)} saved article file(s) on disk  \n\n"
                "This cannot be undone. Continue?"
            )
            confirm_col, cancel_col, _ = st.columns([1, 1, 4])
            with confirm_col:
                if st.button("Yes, delete everything", type="primary", key="t4_confirm_yes"):
                    # 1. Clear all Tab 4 session state
                    for k in list(st.session_state.keys()):
                        if k.startswith("comparison_") or k.startswith("t4_"):
                            if k != "t4_confirm_clear":
                                st.session_state.pop(k, None)
                    st.session_state.pop("comparison_df", None)
                    # 2. Delete all .txt article files in sample_articles/
                    deleted_count = 0
                    for txt_file in SAMPLE_DIR.glob("*.txt"):
                        if txt_file.name != "SAMPLE_TEMPLATE.txt":
                            try:
                                txt_file.unlink()
                                deleted_count += 1
                            except Exception as e:
                                st.error(f"Could not delete {txt_file.name}: {e}")
                    st.session_state["t4_confirm_clear"] = False
                    st.success(f"Cleared cache and deleted {deleted_count} article file(s).")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key="t4_confirm_no"):
                    st.session_state["t4_confirm_clear"] = False
                    st.rerun()

        if run_clicked:
            glove_path = Path(glove_path_str)
            if not glove_path.exists():
                st.error(f"Word vector file not found at {glove_path}.")
            else:
                with st.spinner("Loading resources…"):
                    glove = get_glove(str(glove_path))
                    opinion_text = retriever.get_full_opinion_text()
                    topic_model = get_topic_model(sig, num_topics=8)
                    opinion_topics = document_topic_distribution(opinion_text, topic_model)

                progress = st.progress(0.0, text="Starting…")
                results = []
                for i, art in enumerate(articles):
                    progress.progress(
                        (i + 0.5) / len(articles),
                        text=f"Analyzing {art.outlet} ({art.filename})…",
                    )
                    relevance = check_relevance(
                        news_text=art.body,
                        vectorstore=retriever.vectorstore,
                        opinion_full_text=opinion_text,
                    )
                    row = analyze_one_article(
                        article=art,
                        opinion_text=opinion_text,
                        glove=glove,
                        topic_model=topic_model,
                        opinion_topic_dist=opinion_topics,
                        max_oracle_sentences=oracle_max,
                        relevance_check=relevance,
                        enable_stance=ENABLE_STANCE,
                        enable_judge=ENABLE_JUDGE,
                        case_name=pdf_path.stem.replace("_", " "),
                    )
                    results.append(row)
                    progress.progress((i + 1) / len(articles))
                progress.empty()
                st.session_state["comparison_df"] = pd.DataFrame(results)

        # Render persisted DF
        if "comparison_df" in st.session_state:
            df = st.session_state["comparison_df"].copy()

            if "lean" in df.columns:
                df["Political lean"] = df["lean"].map(LEAN_LABELS).fillna(df["lean"])
            else:
                df["Political lean"] = "unknown"

            display_renames = {
                "outlet": "Outlet",
                "is_relevant": "Article on-topic?",
                "rougeL_f1_vs_oracle": "Faithfulness score",
                "emotionality_news": "Emotional intensity (news)",
                "emotionality_delta": "Emotional shift (vs. ruling)",
                "topic_divergence_from_opinion": "Topic-emphasis difference",
            }
            display_df = df.rename(columns=display_renames)

            st.markdown("### Summary by article")
            display_cols = [
                "Outlet", "Political lean", "Article on-topic?", "Faithfulness score",
                "Emotional intensity (news)", "Emotional shift (vs. ruling)",
                "Topic-emphasis difference",
            ]
            available_cols = [c for c in display_cols if c in display_df.columns]
            st.dataframe(display_df[available_cols], use_container_width=True, hide_index=True)

            # Group means by lean
            st.markdown("### Average scores by political lean")
            numeric_cols = display_df.select_dtypes(include="number").columns.tolist()
            if numeric_cols and "Political lean" in display_df.columns:
                grouped = display_df.groupby("Political lean")[numeric_cols].mean(numeric_only=True).round(3)
                st.dataframe(grouped, use_container_width=True)

            # Faithfulness bar chart
            if "rougeL_f1_vs_oracle" in df.columns:
                st.markdown("### Faithfulness to the ruling, by article")
                fig_bar = go.Figure()
                for lean in df["lean"].unique():
                    sub = df[df["lean"] == lean]
                    fig_bar.add_trace(go.Bar(
                        x=sub["outlet"], y=sub["rougeL_f1_vs_oracle"],
                        name=LEAN_LABELS.get(lean, lean),
                        marker_color=LEAN_COLORS.get(lean, "#9CA3AF"),
                    ))
                fig_bar.update_layout(
                    yaxis_title="Faithfulness score (0 – 1)",
                    xaxis_title="Outlet",
                    showlegend=True, height=400,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Tone fingerprint with toggle
            axis_news_cols = [c for c in df.columns if c.startswith("axis_") and c.endswith("_news")]
            if axis_news_cols:
                st.markdown("### Tone fingerprint by article")

                chart_style_t4 = st.radio(
                    "Chart style",
                    options=["Bar chart", "Radar chart"],
                    horizontal=True,
                    key="t4_chart",
                )

                axis_names = [c.replace("axis_", "").replace("_news", "") for c in axis_news_cols]
                axis_friendly_names = [friendly_axis(a) for a in axis_names]

                if chart_style_t4 == "Bar chart":
                    fig_bar2 = go.Figure()
                    for _, row in df.iterrows():
                        values = [row[c] for c in axis_news_cols]
                        fig_bar2.add_trace(go.Bar(
                            name=f"{row['outlet']} ({LEAN_LABELS.get(row['lean'], row['lean'])})",
                            y=axis_friendly_names, x=values,
                            orientation="h",
                            marker_color=LEAN_COLORS.get(row["lean"], "#9CA3AF"),
                        ))
                    fig_bar2.update_layout(
                        barmode="group",
                        xaxis_title="Score (negative = left side of dimension, positive = right side)",
                        height=500, showlegend=True,
                    )
                    st.plotly_chart(fig_bar2, use_container_width=True)
                else:
                    fig_radar = go.Figure()
                    for _, row in df.iterrows():
                        values = [row[c] for c in axis_news_cols]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values + [values[0]],
                            theta=axis_friendly_names + [axis_friendly_names[0]],
                            fill="toself",
                            name=f"{row['outlet']} ({LEAN_LABELS.get(row['lean'], row['lean'])})",
                            line=dict(color=LEAN_COLORS.get(row["lean"], "#9CA3AF"), width=2),
                            opacity=0.6,
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5])),
                        height=550, showlegend=True,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            # Off-topic warning
            if "is_relevant" in df.columns:
                irrelevant = df[df["is_relevant"] == False]
                if not irrelevant.empty:
                    st.warning(
                        f"{len(irrelevant)} article(s) flagged as possibly off-topic: "
                        f"{', '.join(irrelevant['filename'].tolist())}"
                    )

            st.download_button(
                "💾 Download full results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="article_comparison.csv",
                mime="text/csv",
            )


# ----------------------------------------------------------------------------
# TAB 5: Whose Voice Does the Coverage Echo?
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Break the ruling into separate justice voices")
    st.caption(
        "A Supreme Court ruling has multiple voices — the majority, concurrences, "
        "and dissents. This tab separates them so you can compare their tone "
        "individually and see which voice the news coverage tends to match."
    )

    if st.button("Find the separate voices in this ruling", key="detect_voices"):
        with st.spinner("Reading the ruling and finding the separate voices…"):
            opinion_text = retriever.get_full_opinion_text()
            sections = detect_sections(opinion_text)
            st.session_state["voices_sections"] = sections
            st.session_state["voices_summary"] = voice_summary(sections)

    if "voices_sections" in st.session_state:
        sections = st.session_state["voices_sections"]
        summary = st.session_state["voices_summary"]

        st.markdown(f"### Found {len(sections)} sections")

        if any(s.confidence < 0.5 for s in sections):
            st.warning(
                "Section detection had low confidence on at least one segment. "
                "This means the ruling didn't follow standard formatting cleanly. "
                "Treat the per-voice tone scores as approximations."
            )

        voice_friendly = {
            "majority": "Majority opinion",
            "concurrence": "Concurrence",
            "dissent": "Dissent",
        }

        summary_rows = []
        for i, s in enumerate(summary):
            summary_rows.append({
                "#": i + 1,
                "Type of opinion": voice_friendly.get(s["voice"], s["voice"]),
                "Justice": s["author"],
                "Length (characters)": f"{s['char_count']:,}",
                "Confidence detection": s["confidence"],
                "Opening text": s["preview"][:120],
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if st.button("Compare tone across voices", key="voice_tones"):
            glove_path = Path(glove_path_str)
            if not glove_path.exists():
                st.error(f"Word vector file not found at {glove_path}.")
            else:
                with st.spinner("Loading…"):
                    glove = get_glove(str(glove_path))
                voice_texts = section_texts_by_voice(sections)
                with st.spinner("Computing tone for each voice…"):
                    voice_scores = analyze_tone(voice_texts, glove)
                    voice_emo = {
                        v: emotionality_score(t, glove) for v, t in voice_texts.items()
                    }
                st.session_state["voices_scores"] = voice_scores
                st.session_state["voices_emo"] = voice_emo

        if "voices_scores" in st.session_state:
            voice_scores = st.session_state["voices_scores"]
            voice_emo = st.session_state["voices_emo"]

            st.markdown("### Tone profile by voice")
            df_rows = []
            for axis_name, doc_scores in voice_scores.items():
                row = {"Tone dimension": friendly_axis(axis_name)}
                for voice, score in doc_scores.items():
                    row[voice_friendly.get(voice, voice)] = round(score, 4)
                df_rows.append(row)
            st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

            st.markdown("### Emotional intensity by voice")
            emo_rows = []
            for voice, emo in voice_emo.items():
                emo_rows.append({
                    "Type of opinion": voice_friendly.get(voice, voice),
                    "Emotional intensity": emo["emotionality"],
                })
            st.dataframe(pd.DataFrame(emo_rows), use_container_width=True, hide_index=True)

            chart_style_t5 = st.radio(
                "Chart style",
                options=["Bar chart", "Radar chart"],
                horizontal=True,
                key="t5_chart",
            )

            axis_names_friendly = [friendly_axis(a) for a in voice_scores.keys()]
            voices_present = list(next(iter(voice_scores.values())).keys())

            if chart_style_t5 == "Bar chart":
                fig_v_bar = go.Figure()
                for voice in voices_present:
                    values = [voice_scores[a][voice] for a in voice_scores.keys()]
                    fig_v_bar.add_trace(go.Bar(
                        name=voice_friendly.get(voice, voice),
                        y=axis_names_friendly, x=values,
                        orientation="h",
                    ))
                fig_v_bar.update_layout(
                    barmode="group",
                    xaxis_title="Score (negative = left side of dimension, positive = right side)",
                    height=400, showlegend=True,
                )
                st.plotly_chart(fig_v_bar, use_container_width=True)
            else:
                friendly_voice_scores = {
                    friendly_axis(k): {voice_friendly.get(vn, vn): val for vn, val in v.items()}
                    for k, v in voice_scores.items()
                }
                fig = radar_chart(friendly_voice_scores, title="Tone Profile by Justice Voice")
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption(
    "**Legal Narrative Bias Explorer** | \"Text As Data\" by Sheetal Sood."
)

