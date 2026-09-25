# Amazon ML Challenge 2026 — Business Entity Resolution

## Context

This brief is for a 72-hour hackathon (assessment window 25–27 Sep 2026 IST) on Unstop. Use it to bootstrap the project: set up the repo structure, write the pipeline, and iterate against the training data. Read fully before writing code — the scoring metric changes what "good" looks like (see Scoring).

## Problem

Business records arrive from 3 independent sources. Determine which records across sources describe the same real-world business.

- **Source 1**: clean, deduplicated reference list of businesses (e.g. businesses that signed up directly).
- **Source 2** and **Source 3**: noisy fragments pulled in from external data vendors to enrich the picture of each business. Each vendor has its own formatting conventions.
- **No shared identifier** exists between sources — the *only* usable fields are business name and address. The data is deliberately limited to just these two fields.
- For every Source 1 entity, find all matching records in Source 2 and Source 3. A Source 1 entity may match **many** records, **exactly one**, or **none at all** (a singleton).

The same real business gets described differently by each source: e.g. "Acme Robotics Incorporated" vs. an abbreviated form; one source abbreviates the address, another references a nearby landmark instead of writing it out.

## Pipeline architecture

Comparing every Source 1 record against every Source 2/3 record is infeasible at scale. The task is explicitly a two-stage pipeline:

### 1. Blocking
Sort records into buckets using a cheap key built from name + address, so records likely to match land in the same bucket. Records can group through a similar name *or* through a shared address.

- Blocking is tuned for **recall**, not precision — buckets will also pull in look-alikes: a business with a similar name at a different address, or a different business that happens to share an address. That's expected; the matching model cleans it up next.
- **Blocking sets the ceiling on achievable recall.** A record that's never blocked together with its true match can never be found by any downstream model, however good. Build and validate this stage first, against training ground truth, before investing in the matching model.

### 2. Matching model
Scores each candidate pair from the blocking stage and keeps only true matches, discarding the rest.

## Data

- Two datasets: **training** (all 3 sources + ground truth match labels) and **test** (all 3 sources, no labels — generate predictions for this).
- **All files are tab-separated (.tsv)** — parse with an explicit tab delimiter or columns won't split correctly.
- **Label / submission format**: one row per Source 1 entity. Its ID maps to a comma-separated list of all matching Source 2/3 IDs. Empty list = entity is a singleton (no match). The submission format mirrors this exactly.
- **Constraint**: pure ML challenge — no external databases, APIs, or lookups. Use only the provided data.
- Datasets are visible only to the team leader on Unstop (relevant if working in a team).

## Deliverables

1. **`matching_results.tsv`** — one row per Source 1 entity with predicted matches. This is the **only file scored on the leaderboard**, uploaded throughout the challenge (live leaderboard).
2. **Final archive** (submitted once, at close):
   - `matching_results.tsv` (final version)
   - `candidate_pairs.tsv` — the candidate set produced by the blocking stage (not scored directly, but used to audit blocking quality/recall)
   - Complete, runnable pipeline (code/script/notebook)
   - 1–2 page methodology document describing the approach
   - There's also a **private leaderboard**, revealed after the hackathon, based on the final submission validated against the complete test dataset. Top teams' full packages get manually reviewed before final rankings are confirmed.
3. Run the provided validation script before every submission — a formatting error shouldn't cost a submission attempt.

## Scoring

- **Metric**: macro F0.5 — precision weighted **twice** as heavily as recall.
- Incorrectly merging two different businesses (false positive) costs roughly **2x** what missing a true match (false negative) costs. **When uncertain, prefer not merging.**
- **Singletons matter as much as matches**: correctly predicting an empty match list for a true singleton scores 1.0 on that entity; predicting any match for it scores 0. Correctly identifying non-matches is graded as heavily as finding true matches — don't treat singleton detection as an afterthought.

## Build order (recommended)

1. **EDA on training data.** Look at real matched pairs and real singletons side by side. Characterize the noise per source: abbreviation patterns, landmark-vs-address references, formatting differences, any regional patterns in names/addresses.
2. **Build blocking first, measure its recall ceiling** against training ground truth before writing any matching model — i.e. for each Source 1 entity, check whether its true matches (per ground truth) ended up in the same block. This number upper-bounds your final recall.
3. **Build the matching model** on the candidate pairs blocking produces. Given F0.5, tune the decision threshold toward precision rather than the naive 0.5 default — validate this against training ground truth, don't guess.
4. **Explicitly validate singleton handling** — measure how often true singletons get wrongly assigned a match, since each one flips a perfect 1.0 to a 0.
5. **Run the provided validation script** before every submission.
6. Keep `candidate_pairs.tsv` generation reproducible from the start — it's needed for the final archive, not just an intermediate scratch file.

## Timeline

- Assessment window: 25 Sep 2026, 12:00 AM IST – 27 Sep 2026, 11:59 PM IST
- Duration once started: 2 days 23:59:00 — **the timer does not pause once started**
- Must finish and submit on or before 27 Sep 2026, 11:59 PM IST
- Final submission: 1–2 page approach doc + code/script/notebook, zipped
