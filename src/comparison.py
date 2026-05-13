"""Batch analysis: run every metric across a corpus of news articles."""
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from src.article_store import Article
from src.rouge_metrics import analyze_news_article
from src.semantic import analyze_tone
from src.emotionality import emotionality_score
from src.topic_model import (
    TopicModel, document_topic_distribution, topic_divergence,
)
from src.stance import analyze_stance, stance_to_scalar
from src.llm import llm_judge_all_rubrics, is_ollama_available


def analyze_one_article(
    article: Article,
    opinion_text: str,
    glove: Dict[str, np.ndarray],
    topic_model: Optional[TopicModel] = None,
    opinion_topic_dist: Optional[Dict[int, float]] = None,
    max_oracle_sentences: int = 15,
    relevance_check: Optional[Dict] = None,
    enable_stance: bool = True,
    enable_judge: bool = False,
    case_name: str = "the ruling",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    row: Dict = {
        "filename": article.filename,
        "outlet": article.outlet,
        "lean": article.lean,
        "title": article.title,
        "word_count": article.word_count,
    }

    if relevance_check is not None:
        row["relevance_similarity"] = relevance_check.get("similarity", None)
        row["relevance_term_overlap"] = relevance_check.get("term_overlap", None)
        row["is_relevant"] = relevance_check.get("is_relevant", True)

    if progress_callback:
        progress_callback(f"  ROUGE for {article.outlet}…")
    try:
        rouge_result = analyze_news_article(
            opinion_text=opinion_text,
            news_text=article.body,
            max_oracle_sentences=max_oracle_sentences,
        )
        row["oracle_sentences"] = rouge_result["num_oracle_sentences"]
        row["rouge1_f1_vs_oracle"] = rouge_result["rouge_news_vs_oracle"]["rouge1"]["f1"]
        row["rouge2_f1_vs_oracle"] = rouge_result["rouge_news_vs_oracle"]["rouge2"]["f1"]
        row["rougeL_f1_vs_oracle"] = rouge_result["rouge_news_vs_oracle"]["rougeL"]["f1"]
        row["oracle_summary"] = rouge_result["oracle_summary"][:500]
    except Exception as e:
        row["rouge_error"] = str(e)

    if progress_callback:
        progress_callback(f"  Tone axes for {article.outlet}…")
    try:
        tone = analyze_tone(
            documents={"news": article.body, "opinion": opinion_text},
            glove=glove,
        )
        for axis_name, doc_scores in tone.items():
            row[f"axis_{axis_name}_news"] = doc_scores["news"]
            row[f"axis_{axis_name}_opinion"] = doc_scores["opinion"]
            row[f"axis_{axis_name}_delta"] = doc_scores["news"] - doc_scores["opinion"]
    except Exception as e:
        row["tone_error"] = str(e)

    if progress_callback:
        progress_callback(f"  Emotionality for {article.outlet}…")
    try:
        emo_news = emotionality_score(article.body, glove)
        emo_opinion = emotionality_score(opinion_text, glove)
        row["emotionality_news"] = emo_news["emotionality"]
        row["emotionality_opinion"] = emo_opinion["emotionality"]
        row["emotionality_delta"] = emo_news["emotionality"] - emo_opinion["emotionality"]
    except Exception as e:
        row["emotionality_error"] = str(e)

    if topic_model is not None and opinion_topic_dist is not None:
        if progress_callback:
            progress_callback(f"  Topic distribution for {article.outlet}…")
        try:
            news_topics = document_topic_distribution(article.body, topic_model)
            row["topic_divergence_from_opinion"] = topic_divergence(
                news_topics, opinion_topic_dist
            )
            top3 = sorted(news_topics.items(), key=lambda x: -x[1])[:3]
            row["top_topics"] = "; ".join(f"T{tid}({prob:.2f})" for tid, prob in top3)
        except Exception as e:
            row["topic_error"] = str(e)

    if enable_stance:
        if progress_callback:
            progress_callback(f"  Stance for {article.outlet}…")
        try:
            stance_result = analyze_stance(article.body)
            if stance_result is not None:
                row["stance_predicted"] = stance_result["predicted_label"]
                row["stance_confidence"] = stance_result["confidence"]
                row["stance_scalar"] = stance_to_scalar(stance_result)
        except Exception as e:
            row["stance_error"] = str(e)

    if enable_judge and is_ollama_available():
        if progress_callback:
            progress_callback(f"  LLM judge for {article.outlet}…")
        try:
            judge_results = llm_judge_all_rubrics(article.body, case_name=case_name)
            for rubric_key, result in judge_results.items():
                row[f"judge_{rubric_key}"] = result["score"]
                row[f"judge_{rubric_key}_reason"] = result["reasoning"][:200]
        except Exception as e:
            row["judge_error"] = str(e)

    return row


def analyze_corpus(
    articles: List[Article],
    opinion_text: str,
    glove: Dict[str, np.ndarray],
    topic_model: Optional[TopicModel] = None,
    relevance_checker=None,
    vectorstore=None,
    enable_stance: bool = True,
    enable_judge: bool = False,
    case_name: str = "the ruling",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    opinion_topic_dist = None
    if topic_model is not None:
        opinion_topic_dist = document_topic_distribution(opinion_text, topic_model)

    rows = []
    for i, article in enumerate(articles):
        if progress_callback:
            progress_callback(i / len(articles), f"Analyzing {article.outlet}")

        relevance = None
        if relevance_checker is not None and vectorstore is not None:
            try:
                relevance = relevance_checker(
                    news_text=article.body,
                    vectorstore=vectorstore,
                    opinion_full_text=opinion_text,
                )
            except Exception:
                relevance = None

        row = analyze_one_article(
            article=article,
            opinion_text=opinion_text,
            glove=glove,
            topic_model=topic_model,
            opinion_topic_dist=opinion_topic_dist,
            relevance_check=relevance,
            enable_stance=enable_stance,
            enable_judge=enable_judge,
            case_name=case_name,
        )
        rows.append(row)

    if progress_callback:
        progress_callback(1.0, "Done")

    return pd.DataFrame(rows)
