"""LDA topic modeling with Jensen-Shannon divergence for topic comparison."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from gensim import corpora
from gensim.models import LdaModel

from src.utils import clean_text


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "for", "to", "in",
    "on", "at", "by", "with", "as", "is", "was", "were", "be", "been",
    "being", "are", "this", "that", "these", "those", "it", "its", "we",
    "they", "their", "them", "he", "she", "his", "her", "him", "i", "you",
    "your", "our", "us", "from", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "not",
    "no", "yes", "so", "than", "then", "there", "here", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "all", "each",
    "every", "any", "some", "such", "into", "out", "up", "down", "over",
    "under", "more", "most", "other", "another", "same", "much", "very",
    "also", "only", "even", "just", "now", "well", "still", "however",
    "thus", "though", "while", "after", "before", "again", "because",
    "between", "during", "through", "above", "below", "against", "among",
    "within", "without", "upon", "shall", "page", "see",
}


@dataclass
class TopicModel:
    model: LdaModel
    dictionary: corpora.Dictionary
    num_topics: int
    topic_terms: Dict[int, List[Tuple[str, float]]]

    def topic_label(self, topic_id, n_words=4):
        terms = self.topic_terms[topic_id][:n_words]
        return ", ".join(w for w, _ in terms)


def _preprocess(text, min_token_len=3):
    import re
    cleaned = clean_text(text).lower()
    tokens = re.findall(r"[a-z]+", cleaned)
    return [t for t in tokens if len(t) >= min_token_len and t not in STOPWORDS]


def fit_topics(documents, num_topics=8, passes=10, random_state=42):
    tokenized = [_preprocess(d) for d in documents if d.strip()]
    tokenized = [t for t in tokenized if len(t) >= 5]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    bow_corpus = [dictionary.doc2bow(t) for t in tokenized]

    lda = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha="auto",
        eta="auto",
    )
    topic_terms = {i: lda.show_topic(i, topn=10) for i in range(num_topics)}
    return TopicModel(
        model=lda, dictionary=dictionary,
        num_topics=num_topics, topic_terms=topic_terms,
    )


def document_topic_distribution(text, tm, threshold=0.0):
    tokens = _preprocess(text)
    bow = tm.dictionary.doc2bow(tokens)
    dist = tm.model.get_document_topics(bow, minimum_probability=threshold)
    full = {i: 0.0 for i in range(tm.num_topics)}
    for tid, prob in dist:
        full[tid] = float(prob)
    return full


def topic_divergence(p, q):
    """Square root of Jensen-Shannon divergence (proper metric in [0, 1])."""
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_arr = np.array([p.get(k, 0) for k in keys]) + 1e-12
    q_arr = np.array([q.get(k, 0) for k in keys]) + 1e-12
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * np.log(p_arr / m))
    kl_qm = np.sum(q_arr * np.log(q_arr / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(js))
