# ⚖️ The Legal Narrative Bias Explorer

A Streamlit app that compares news coverage of a legal case to the actual judicial opinion across multiple NLP dimensions.

Built for NYU MSDS *Text As Data* (DS-GA 1015) by Sheetal Sood.

---

## What the app does

The app takes two kinds of text:

1. a Supreme Court opinion PDF, treated as the primary source text; and  
2. one or more news articles, treated as secondary coverage of that ruling.

It then compares the article against the opinion across five measurements. The goal is not to label an article as simply "true" or "biased." Instead, the app gives computational signals for how closely the article follows the ruling, how its tone shifts, and which judicial voice it resembles.

---

## What it measures

| Tab | Question | Method |
|---|---|---|
| 1. Search the Opinion | What did the court actually say? | LEGAL-BERT embeddings + ChromaDB retrieval with page citations |
| 2. How Faithful is the Coverage? | How closely does the article match the ruling? | Greedy Oracle sentence extraction + ROUGE-1/2/L |
| 3. How Did the Framing Change? | Did the article sound more emotional, concrete, or active than the ruling? | GloVe semantic axes + Gennaro & Ash emotionality score |
| 4. Compare News Articles | How do saved articles differ by outlet or political lean? | Batch comparison of faithfulness, tone, topic difference, and relevance |
| 5. Whose Voice Does the Coverage Echo? | Does the coverage sound closer to the majority, concurrence, or dissent? | Regex-based judicial voice detection + per-voice tone profile |

The final reported results use deterministic NLP pipelines only. Older optional modules for stance detection or LLM-based judging may exist in the source folder, but they are disabled in the final app and are not used in the results.

---

## Quick start

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# 2. Add a court opinion PDF
mkdir -p data/raw
# Example: place dobbs_opinion.pdf or another Supreme Court PDF inside data/raw/

# 3. Add GloVe vectors for tone analysis
curl -L -o glove.6B.zip "https://nlp.stanford.edu/data/glove.6B.zip"
unzip glove.6B.zip glove.6B.300d.txt -d data/
rm glove.6B.zip

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How to use the app

1. Upload or select a court opinion PDF in the sidebar.
2. Use **Search the Opinion** to ask questions about the legal ruling.
3. Paste a news article into **How Faithful is the Coverage?** to compute a faithfulness score.
4. Paste the same article into **How Did the Framing Change?** to compare tone.
5. Save articles with outlet name and political lean.
6. Use **Compare News Articles** to compare all saved articles together.
7. Use **Whose Voice Does the Coverage Echo?** to split the opinion into majority, concurrence, and dissent sections.

---

## Project structure

```text
legal-narrative-bias/
├── README.md
├── METHODOLOGY.md
├── DEFENSE_CHEATSHEET.md
├── requirements.txt
├── app.py
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── rag.py
│   ├── rouge_metrics.py
│   ├── semantic.py
│   ├── emotionality.py
│   ├── topic_model.py
│   ├── relevance.py
│   ├── article_store.py
│   ├── comparison.py
│   └── justice_voice.py
├── scripts/
│   └── ingest.py
├── sample_articles/
└── data/
    ├── raw/
    ├── chroma_db/
    └── glove.6B.300d.txt
```

---

## Notes on interpretation

Scores should be treated as computational proxies, not final judgments of truth or bias. A low ROUGE score can happen when an article faithfully paraphrases the ruling using different words. Similarly, a higher emotionality score does not automatically mean the article is wrong; it only means the article uses language that is more emotionally framed relative to the legal opinion.

---

## References

- Blei, Ng, and Jordan (2003). *Latent Dirichlet Allocation.*
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of law school.*
- Gennaro and Ash (2022). *Emotion and Reason in Political Language.*
- Kozlowski, Taddy, and Evans (2019). *The Geometry of Culture.*
- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
- Nallapati, Zhai, and Zhou (2017). *SummaRuNNer.*
- Pennington, Socher, and Manning (2014). *GloVe.*
