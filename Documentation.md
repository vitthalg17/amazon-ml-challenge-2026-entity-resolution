# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

The system is a three-stage entity-resolution pipeline that runs on a 16 GB laptop CPU:
1. **Per-country multi-channel TF-IDF blocking**, pruned by a small LightGBM model.
2. **A LightGBM pair classifier** with about 110 name, address, house-number, legal-form and candidate-context features.
3. **A cluster-consistency re-scorer.** It checks each candidate against the other records already assigned to the same Source-1 entity.

Every Source-2/3 record goes to at most one Source-1 entity. The final decision rule is tuned directly for macro F0.5 on a large held-out part of the training data. No external data, APIs or pretrained models are used.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.**
  - Train: 2.2 M Source-1 entities, about 10.3 M Source-2/3 records.
  - Test: 1.73 M Source-1 entities and 9.97 M Source-2/3 records.
  - A full cross join is impossible, so blocking must reduce roughly 10¹³ possible pairs to about 1.7 candidates per record.
- **Countries.** Train contains the US and India. Test also contains **France**, which never occurs in training. Every model input is therefore country-agnostic text similarity, not a country-specific rule. Matching never crosses countries.
- **Noise patterns in names:**
  - abbreviations and legal-form variation (`Pvt Ltd` / `Private Limited`, `Inc`, `LLC`);
  - word re-ordering;
  - typos;
  - location suffixes glued to the name ("Cafe Coffee Day – Koramangala");
  - Indic-script copies of Latin names (Devanagari, Tamil, …).
- **Noise patterns in addresses:**
  - abbreviations (`St`/`Street`, `Rd`/`Road`);
  - state names vs. codes;
  - missing fields;
  - re-ordered components;
  - house numbers written differently.
  - A sizeable share of records has **no address at all**. For these, name + city is the only evidence.
- **Hard negatives (distractors).** Many Source-2/3 records have *no* true match but look like one: another branch of the same chain, the same name at a different house number, or the same name with a different legal form. Under F0.5 a wrong merge costs more than a miss, so precision on these distractors drives the score.
- **Many-to-one.** One Source-1 entity can have several Source-2/3 copies (up to 11 in training), while each Source-2/3 record belongs to at most one entity.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier, with a cluster-consistency re-scoring stage (hybrid).

**Core Innovation:**
- **Memory-bounded full-density training.** Blocking, feature building and both models run over *all* training records (the same density as test) on a laptop. Everything is streamed in chunks and spilled to disk, and each step and each feature part runs in its own short-lived process. A watchdog stops a job before RAM runs out, and every step resumes from its finished parts.
- **Density matching.** With it, candidate-context features ("how does this candidate compare with the record's other candidates / the entity's other candidates") and the per-entity expected-F0.5 decision rule behave identically in training and in test.
- **Stage 3.** It adds cluster evidence: a distractor usually disagrees with the entity's already-matched copies (house number, legal form, street), while a noisy true copy agrees with most of them.

---

## 3. Candidate Generation (Blocking)

Blocking is done **per country**. The Source-1 records of the country are the index, and the Source-2/3 records are the queries (streamed 50 k at a time).

**Blocking keys used:**

| Channel | Representation | Top-K per record |
|---|---|---|
| `name_word` | TF-IDF of name word uni+bigrams plus an order-free name key | 20 |
| `name_char` | TF-IDF of character 4-grams (`char_wb`) of the name, very common grams dropped | 10 |
| `addr` | TF-IDF of address word uni+bigrams | 20 |
| `combo` | TF-IDF of name + address together | 20 |
| `exact` | identical order-free core name, for names shared by ≤ 50 entities | all |
| `exact_addr` | identical core name shared by 51–3000 entities (chains), ranked by address cosine | 5 |

- Retrieval is a sparse matrix product with top-K selection (`sparse_dot_topn`).
- Very common query features are removed on the *query* side only. They carry little IDF weight but dominate the cost.
- Full vectors are kept for the cosine features.

**Normalization before blocking:**
- Unicode → ASCII transliteration (`anyascii`).
- A small **Indic → Latin token dictionary learned from the training ground truth** (aligned tokens of matched name pairs; 566 entries). It is cross-fitted on train so that no record sees a dictionary built from its own label.
- Hand-written lists of legal forms, address abbreviations and US/Indian state names.
- Extraction of a core name (legal form and location suffix removed), the street and the house number.

**Stage-1 pruner:**
- A LightGBM model on the retrieval signals (the per-channel ranks and cosines) keeps, per record, the top 10 candidates with probability ≥ 1e-4.
- It is trained on a 2 % slice of training records, cross-fitted: two models on disjoint halves, and each training record is pruned by the model that did not see it.
- Records of countries unseen in training (France) use the same rule.

**Candidate pairs generated:**
- Test: 16,830,060 pairs for 9.97 M records, about 1.7 per record.
- Train (full density): [v7 number].

**How true matches were kept:**
- The channels are complementary: character n-grams catch typos and transliteration, word/bigram channels catch re-ordering, the address channel catches renamed businesses, and the exact-name channels catch chains.
- Blocking recall is measured against the training ground truth after every change (`eval_blocking.py`). Pair recall is **98.5 %** of all true pairs [v7 number].
- The remaining misses are mostly records with no address that share a very common name, where no text signal can separate them.

---

## 4. Matching Model

**Stage 2: pair classifier (LightGBM, about 110 features).**

**Name features:**
- `rapidfuzz` ratio, partial ratio, token-sort and token-set ratios, and Jaro-Winkler, on the normalized, core and alternative (transliterated) names.
- Jaccard and IDF-weighted token overlap (shared and unmatched token weight on each side).
- The frequency of the core name in Source 1 (common names are weak evidence).
- A comparison of the location suffix of the name.

**Legal-form features:**
- Legal-form tokens on each side, whether they agree, and whether one side has none.

**Address features:**
- Token-set and ratio similarity of the full address and the street.
- Jaccard and IDF overlap.
- Number tokens: overlap, first-number edit distance, and house-number equality, edit distance, prefix match, containment and missingness.
- Empty-address indicators.

**Retrieval features:**
- The four channel cosines, their sum, the channel ranks and the stage-1 probability.

**Context features:**
- For each record, the gap of every similarity to the best candidate of the same Source-2/3 record and of the same Source-1 entity.
- Rank among those candidates, number of candidates, and the margin to the second-best candidate.

**Training:**
- LightGBM (learning rate 0.08, 127 leaves, up to 3000 rounds with early stopping) with 4 folds grouped by Source-1 entity.
- Early stopping uses a different 10 % entity slice per fold.
- The model is fitted on 30 % of the training Source-2/3 records.
- The other 70 % is a genuine hold-out, scored by the fold-model average. Every training pair therefore has an honest probability: out-of-fold or never fitted.

**Stage 3: cluster-consistency re-scoring (LightGBM, 19 features).**
- An entity's "members" are the records whose best stage-2 match is that entity with p ≥ 0.5.
- For each candidate pair, the candidate is compared with those members (excluding itself) on:
  - name, address and street similarity (max/mean);
  - house-number agreement;
  - legal-form agreement;
  - number of members and their probability mass.
- These features are combined with the stage-2 probability and its context: gap to the record's best candidate, second-best, and the entity's total probability.
- Stage 3 uses the same folds and hold-out as stage 2. Calibration keeps stage 3 only if it scores higher on the hold-out.

**Model type:** Gradient-boosted trees (LightGBM), two stages.

**Threshold selection method:**
- Each Source-2/3 record keeps only its highest-scoring Source-1 entity.
- Two decision rules are searched on the held-out training probabilities with the exact leaderboard metric (macro F0.5 over *all* Source-1 entities, including those with no match):
  - (a) a global probability threshold;
  - (b) a per-entity rule that picks, for each Source-1 entity, the set of candidates that maximizes its expected F0.5 (with a tuned shift).
- The better rule, and the better of stage 2 and stage 3, wins.
- **Calibrating for the test mix.** Test has the same number of true matches per Source-1 entity as train:
  - train: 3.46 per entity; test: 3.33 predicted;
  - 5.6 % of entities have no match in both.
  - But test has 5.75 Source-2/3 records per entity instead of 4.68, i.e. about **1.89× as many distractor records**.
  - The selection metric therefore counts every loss caused by a distractor 1.89 times. Per entity: F0.5 without its distractor false positives, minus w × the loss those false positives cause.
  - The ratio comes only from row counts of the competition files. It moves the decision towards precision where the test mix demands it.
- Safety guards for the test-only country (France): threshold + 0.1, at most 11 records per entity (the training maximum), and per-source caps learned from training.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), held-out validation:** [v7 number]. Leaderboard history: 0.9545 → 0.959 → 0.968 → [v7].
- **Common false positives (wrong merges):**
  - Distractors of the same chain or name at another address.
  - Legal-form swaps (`X Pvt Ltd` vs `X LLP`).
  - The same name with house numbers that differ by one digit.
- **Common false negatives (missed matches):**
  - About 60 % are blocking misses. Of those, about 42 % are records without an address whose name is shared by many entities; they are unrecoverable from text.
  - The rest are true pairs scored just below the threshold (heavy abbreviation, transliteration plus typo), or records "won" by a sibling entity with a near-identical name.

---

## 6. Conclusion

Careful normalization, complementary retrieval channels and a strong gradient-boosted pair model get most of the way. The biggest late gains came from:
- training at the **same candidate density as test**, which only became possible on a laptop through streaming, spilling and per-part processes;
- **cluster-level evidence** that separates distractors from true copies.

Lesson: in an F0.5 entity-resolution task, the distractors, not the easy matches, decide the ranking, so features that expose *disagreement* (house number, legal form, sibling consistency) matter most.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `README.md`
- `requirements.txt` (polars, scikit-learn, sparse_dot_topn, rapidfuzz, lightgbm, anyascii, numpy, scipy, psutil)
- `src/`:
  - `pipeline.py`: entry point; runs every step in its own process.
  - `ingest.py`, `prep.py`, `normalize.py`, `translit.py`: parsing and normalization.
  - `blocking.py`, `stage1.py`, `eval_blocking.py`: candidate generation.
  - `features.py`, `match.py`: stage-2 features, model, calibration, prediction and outputs.
  - `cluster.py`: stage 3.
  - `metrics.py`: macro F0.5 exactly as defined by the challenge.

Reproduce:

```bash
pip install -r requirements.txt
cd src
python pipeline.py --train-pct 100 --test-pct 100 --fit-pct 30   # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

On a 16 GB machine, wrap the run in `tools/run_guarded.py` (RAM watchdog). Any step can be resumed with `--steps`.

### B. Additional Results

| Version | Change | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v1 | TF-IDF blocking + LightGBM, train 30 % | 0.9686 | 0.9545 |
| v4 | France fixes, 2000 rounds, guards | 0.9697 | 0.959 |
| v6 | deeper blocking (exact/exact_addr channels), legal and house-number features, caps | 0.9780 | 0.968 |
| v7 | full-density training (100 %), S1-context features, entity rule, stage 3 | [v7] | [v7] |
