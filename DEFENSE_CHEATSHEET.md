# Defense Cheat Sheet

These are the main questions I asked myself while building the Legal Narrative Bias Explorer, along with the answers I would give during a project discussion or defense. I wrote this as a preparation sheet, not as a formal report section.

---

### Q1. What is the main goal of my project?

The goal is to compare news coverage of Supreme Court rulings to the actual judicial opinions. I am not trying to prove that one outlet is "good" and another outlet is "bad." I am measuring several textual differences: whether the article is about the same ruling, how closely it matches the legal text, whether the tone shifts, whether the topic emphasis changes, and whether the article sounds closer to the majority, concurrence, or dissent.

---

### Q2. Why did I use LEGAL-BERT instead of a general sentence model?

I used LEGAL-BERT because the source documents are legal opinions, not ordinary news or casual text. Supreme Court opinions contain terms like *stare decisis*, *rational basis review*, *strict scrutiny*, *substantive due process*, and *equal protection*. A legal-domain model is more appropriate for retrieving passages from legal text because it was trained on legal language. A general model may still work, but LEGAL-BERT better matches the domain of the primary source.

---

### Q3. How did I split the opinion into chunks?

I used a hybrid sentence + token-aware chunker. First, the PDF text is split into sentences using NLTK, with a regex fallback if needed. Then I pack complete sentences into chunks of up to 400 LEGAL-BERT tokens, with a 2-sentence overlap between neighboring chunks. This keeps the chunks within the model's context window while avoiding mid-sentence cuts. That matters because a legal sentence can contain the holding, exception, and reasoning all together.

---

### Q4. Why not just compare a news article directly to the full opinion?

Direct comparison is unfair because of length mismatch. A news article may be 300 to 800 words, while a Supreme Court opinion can be tens of thousands of words. If I compare the article to the entire opinion, recall will be mechanically low because the article cannot possibly contain most of the ruling. That is why I use a Greedy Oracle baseline: it finds the small set of ruling sentences that best match the article, then compares the article to those sentences instead of to the full opinion.

---

### Q5. What is the Greedy Oracle doing in simple terms?

The Greedy Oracle asks: "Which sentences from the ruling would create the closest possible extractive summary of this news article?" It picks one ruling sentence at a time, each time choosing the sentence that increases ROUGE-L overlap the most. The output is not a human-written summary; it is a best-matching set of source sentences. This gives a fairer baseline for measuring how much the article follows or departs from the ruling's wording.

---

### Q6. Why did I use ROUGE?

ROUGE is useful because it measures word overlap between the news article and the ruling sentences. ROUGE-1 checks single-word overlap, ROUGE-2 checks word-pair overlap, and ROUGE-L checks the longest matched word sequence. I used ROUGE because I wanted a transparent measure of lexical faithfulness. A semantic metric like BERTScore could be useful in future work, but ROUGE makes it easier to see when an article is quoting or closely paraphrasing the ruling.

---

### Q7. Why does a low faithfulness score not automatically mean the article is wrong?

A low score can mean several things. It can mean the article is using different framing, but it can also mean the article is faithfully paraphrasing the ruling in different words. ROUGE is a surface-overlap metric, so it rewards shared wording. That is why I describe the score as a computational proxy, not a final judgment of truth or bias.

---

### Q8. How does the tone analysis work?

For tone, I use GloVe word vectors. I define axes such as Reason vs. Emotion, Abstract vs. Concrete, and Passive vs. Active using anchor words at both ends. Each document is represented as the average of its word vectors, and then I project that document vector onto each axis. This gives a tone profile for the ruling and for the news article, so I can compare whether the article sounds more emotional, more concrete, or more active than the legal opinion.

---

### Q9. Why did I use GloVe for tone instead of LEGAL-BERT?

The semantic-axis method needs one stable vector per word, because the axes are built from anchor words like *logic*, *evidence*, *fear*, and *outrage*. GloVe gives one type-level vector for each word, which makes it suitable for this kind of geometry. LEGAL-BERT is contextual, so the vector for a word changes depending on the sentence. That is useful for retrieval, but less direct for simple anchor-word axis construction.

---

### Q10. What is the emotionality score?

The emotionality score follows the idea from Gennaro and Ash. I compare a document vector to an emotion-word centroid and a reason-word centroid. The score is calculated as cosine similarity to emotion words minus cosine similarity to reason words. Higher values mean more emotional language; lower or negative values mean more analytical language. In my results, the court opinions usually have negative scores, which makes sense because legal writing is generally analytical.

---

### Q11. What does topic modeling add?

Topic modeling helps measure topic emphasis. Two articles can both be related to the same case but focus on different parts of it. For example, one article may emphasize constitutional doctrine, while another emphasizes political consequences. I use LDA to learn topics from the opinion chunks and then compare the article's topic distribution to the ruling's topic distribution using Jensen-Shannon divergence.

---

### Q12. Why Jensen-Shannon divergence instead of KL divergence?

Jensen-Shannon divergence is symmetric and bounded, which makes it easier to interpret. KL divergence can behave badly when one distribution gives zero probability to something that appears in the other distribution. JS divergence avoids that issue and gives a cleaner distance-like measure between topic distributions.

---

### Q13. How does the app identify majority, concurrences, and dissents?

The app uses regex patterns based on standard Supreme Court slip-opinion headers, such as `JUSTICE X, dissenting` or `JUSTICE X, concurring`. Text before the first detected separate opinion is treated as the majority opinion. This works well for standard SCOTUS opinions, but it is not perfect because PDF extraction can be messy and some opinions have unusual formats.

---

### Q14. What are the biggest limitations of the project?

The main limitations are: PDF text extraction can be noisy, ROUGE misses faithful paraphrases, semantic axes depend on hand-picked anchor words, and justice-voice detection is regex-based. Also, the project is associational, not causal. It can show that coverage differs by tone, faithfulness, or topic emphasis, but it does not prove why an outlet framed the case in a particular way.

---

### Q15. What would I improve with more time?

With more time, I would test more Supreme Court cases and more articles per case, add human annotations to validate the faithfulness and tone scores, and compare ROUGE with a semantic metric like BERTScore. I would also improve the justice-voice detector for unusual opinions and experiment with data-driven tone axes instead of only using hand-picked anchor words.

---

## Short project explanation

My app compares news coverage of a Supreme Court case to the actual opinion. It uses legal-text retrieval to find relevant passages, Greedy Oracle plus ROUGE to measure faithfulness, word-embedding methods to measure tone shift, LDA plus Jensen-Shannon divergence to compare topic emphasis, and regex-based opinion decomposition to compare news tone against majority, concurring, and dissenting voices.

## Short limitations line

The scores are computational proxies, not final judgments of truth or bias. The app is useful for comparing patterns across articles, but the results should be interpreted alongside the actual article text and the legal opinion.
