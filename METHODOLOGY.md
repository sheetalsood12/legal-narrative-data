# Methodology

## 1. Research Design

This project measures how news coverage of a Supreme Court decision differs from the underlying judicial opinion. The system reports five NLP measurements: retrieval over the opinion, faithfulness to the ruling, tonal and framing shift, topic-emphasis comparison, and judicial voice alignment. Each measurement is computed by a separate pipeline so that a difference on one dimension does not force a difference on another.

The unit of analysis is a (judicial opinion, news article) pair. The judicial opinion is treated as the primary source text, while news articles are treated as secondary texts whose relationship to the ruling is what gets measured. The system is case-agnostic; the test case used during development was *Dobbs v. Jackson Women's Health Organization* (2022).

## 2. Retrieval Over the Opinion

The opinion PDF is parsed page by page using `pypdf`, with page numbers retained so that retrieved passages can be cited back to the original document.

The text is then split using a hybrid sentence- and token-aware chunker. Sentences are first identified using NLTK's punkt tokenizer (with a regex fallback when punkt is unavailable). Sentences are then packed into chunks of up to 400 tokens, with a 2-sentence overlap between neighboring chunks. Token counts are measured with LEGAL-BERT's own tokenizer so that the chunk size matches what the embedding model actually consumes. Chunks under 30 tokens are dropped to filter out page footers and running headers. This approach helps ensure that chunks are not cut mid-sentence, which matters for legal reasoning because a single sentence can carry the full holding.

Each chunk is embedded using LEGAL-BERT (`nlpaueb/legal-bert-base-uncased`), pre-trained on roughly 12 GB of legal text (Chalkidis et al., 2020). Token embeddings are mean-pooled to a 768-dimensional vector per chunk and L2-normalized. Embeddings are persisted in a local ChromaDB collection. Each opinion gets its own collection, named with a slug of the filename plus a 12-character SHA-1 hash of the file bytes, so the collection rebuilds automatically when the PDF changes.

Retrieval uses cosine similarity, returning the top-k matching chunks for a query along with their page numbers. The final version uses retrieval-only RAG. No LLM is used to generate answers, which avoids the risk of the model inventing legal claims or producing plausible but unsupported summaries.

## 3. Faithfulness to the Ruling

### 3.1 Why naive ROUGE is misleading

Direct ROUGE comparison between a news article and the full opinion is not interpretable because of the length mismatch. A 300-word article will, almost by construction, score low recall against a 50,000-word ruling, since most of the ruling's vocabulary cannot fit in such a short text. We need a length-matched baseline.

### 3.2 Greedy Oracle baseline

The fix is the Greedy Oracle summary (Nallapati et al., 2017): an extractive summary designed to be as close as possible to the news article at roughly the article's length. Given the opinion's sentences and the news article as target, the algorithm iteratively picks the sentence that produces the largest gain in ROUGE-L overlap with the article, stopping when no remaining sentence produces a gain above 1e-4 or after 25 sentences.

To keep the algorithm tractable on long opinions (the Dobbs ruling has roughly 4,000 sentences), candidates are pre-filtered by initial ROUGE-1 overlap and only the top 200 are passed to the greedy loop. A 90-second timeout caps any pathological cases. This speed-up is reasonable because sentences with near-zero initial word overlap are unlikely to be selected in the greedy stage.

### 3.3 Reported metrics

Three comparisons are reported, each using ROUGE-1, ROUGE-2, and ROUGE-L (Lin, 2004) with stemming enabled:

1. News article vs. full opinion: mechanically low recall, included for context.
2. News article vs. Greedy Oracle summary: the headline faithfulness score.
3. Greedy Oracle vs. full opinion: a sanity check that the oracle is small relative to the ruling.

The ROUGE-L F1 from comparison (2) is reported in the app as the "Faithfulness score."

## 4. Tonal and Framing Shift

### 4.1 Semantic axes

Following Kozlowski, Taddy, and Evans (2019), three tonal axes are constructed in word-embedding space. For each axis, two anchor word lists (8 to 10 words per pole) are chosen by hand. The axis vector is the difference of the centroids of the two pole vectors, then unit-normalized. A document is represented as the mean of its in-vocabulary content-word vectors (minimum 3 characters) and projected onto each axis.

The three axes are *Reason vs. Emotion*, *Abstract vs. Concrete*, and *Passive vs. Active*. Anchor lists are stored in `src/config.py` and chosen to capture the typical contrast between legal register and journalistic register on each dimension. Word vectors come from pre-trained GloVe 6B 300d (Pennington et al., 2014).

### 4.2 Emotionality scalar

Following Gennaro and Ash (2022), a single emotionality score is computed as:

> emotionality(d) = cos(v_d, A_emotion) − cos(v_d, A_reason)

where `A_emotion` is the centroid of an emotion-word list (roughly 30 words including *anger*, *fear*, *joy*, *grief*) and `A_reason` is the centroid of a reason-word list (roughly 30 words including *analysis*, *evidence*, *logic*, *doctrine*). This produces one comparable number per document. Higher means more emotional, lower or negative means more analytical.

The two methods are complementary: the Kozlowski axes give a tone profile across multiple dimensions, while the Gennaro-Ash scalar collapses emotion vs. reason into a single number that is easy to compare across articles.

## 5. Topic Modeling

Latent Dirichlet Allocation (Blei, Ng, and Jordan, 2003) is fit using gensim over the chunked opinion. The model learns 8 topics with `alpha="auto"`, `eta="auto"`, 10 passes, and `random_state=42`. Preprocessing keeps alphabetic tokens of at least 3 characters and drops a custom stopword list. The dictionary filters words appearing in fewer than 2 chunks or in more than 80% of chunks.

For each news article and the opinion itself, the LDA model produces a probability distribution over the 8 topics. Topic-emphasis difference is measured using the square root of the Jensen-Shannon divergence between the two distributions, which is a proper metric in [0, 1]. JS divergence is symmetric and bounded, which is why it is preferred here over KL divergence.

## 6. Judicial Voice Alignment

A Supreme Court ruling typically contains several voices: a majority opinion, one or more concurrences, and one or more dissents. The system splits an opinion into these voices using regex patterns calibrated for SCOTUS slip opinion conventions:

- `JUSTICE [NAME], (with whom..., )?dissenting`
- `JUSTICE [NAME], (with whom..., )?concurring(...in part)?`
- `CHIEF JUSTICE [NAME], ...`

Everything before the first detected header is labeled as the majority opinion. Near-duplicate headers within 200 characters are deduplicated to handle the fact that section headers often repeat across page breaks in slip-opinion PDFs.

Each detected section has a confidence score: 0.85 for sections found by header matching, 0.3 for the fallback case where no headers are detected. The tone pipeline (axes plus emotionality) is then applied to each voice independently. This allows the comparison: does a particular news article's tone profile look more like the majority opinion or more like the dissent?

This step is best-effort. PDF text extraction is messy, and atypical opinion formats (per curiam, summary affirmance) may collapse to a single majority segment.

## 7. Relevance Check

To prevent meaningless analysis on accidentally off-topic articles, the system runs a relevance check before scoring an article. Two signals are combined:

1. Cosine similarity between the article's LEGAL-BERT embedding and the centroid of the opinion's chunk embeddings.
2. Case-name term overlap: capitalized 1-to-4-word phrases that occur at least twice in the opinion, filtered against a small blocklist (United States, Supreme Court, Constitution, etc.), then checked for presence in the article.

An article is flagged "relevant" if similarity is at least 0.60 or term overlap is at least 0.5. Articles below threshold get a warning verdict alongside the analysis output, instead of being silently scored.

## 8. Reproducibility

All model names, chunking parameters, anchor word lists, stopword lists, and similarity thresholds are pinned in `src/config.py`. Topic modeling uses `random_state=42`. LEGAL-BERT and GloVe inference are deterministic on CPU, and the Greedy Oracle is a deterministic greedy argmax with stable tie-breaking by sentence index. In practice, running the pipeline on the same inputs produces highly similar outputs across runs; minor variation may appear from underlying library version differences but the headline metrics are stable.

## 9. Limitations

- PDF extraction noise (column artifacts, footnote text, hyphenated line breaks) survives into the chunked text and can slightly perturb embeddings and ROUGE scores.
- The Kozlowski axes depend on hand-picked anchor words. Different anchor lists would produce different projections. The lists used here are published in `src/config.py` for replication.
- Justice-voice segmentation is regex-based and works well on standard SCOTUS slip opinions but degrades on atypical formats.
- LEGAL-BERT was pre-trained on legal text from EU and US sources, not specifically on US journalism. Cross-domain embedding may understate similarity for journalistic paraphrase.
- ROUGE is a surface-overlap metric. A faithful paraphrase that uses different vocabulary will receive a low score even though it conveys the same content.
- The analysis is associational, not causal. The system measures relationships between framing and outlet, not whether one causes the other.

## References

Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.

Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., and Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of law school. *Findings of EMNLP 2020*.

Gennaro, G., and Ash, E. (2022). Emotion and Reason in Political Language. *The Economic Journal*, 132(643), 1037–1059.

Kozlowski, A. C., Taddy, M., and Evans, J. A. (2019). The Geometry of Culture: Analyzing the Meanings of Class through Word Embeddings. *American Sociological Review*, 84(5), 905–949.

Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Proceedings of the ACL Workshop: Text Summarization Branches Out*, 74–81.

Nallapati, R., Zhai, F., and Zhou, B. (2017). SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents. *AAAI 2017*.

Pennington, J., Socher, R., and Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP 2014*, 1532–1543.