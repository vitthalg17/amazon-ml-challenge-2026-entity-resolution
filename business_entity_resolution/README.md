# Business Entity Resolution — Amazon ML Challenge 2026

Three-stage pipeline:
1. **Blocking**: multi-channel TF-IDF retrieval plus exact-name channels, with a stage-1 pruner.
2. **A LightGBM matching model.**
3. **A cluster-consistency re-scorer.** It compares each candidate with the records already matched to the same Source 1 entity.

Each Source 2/3 record goes to at most one S1 entity. The decision rule is tuned for F0.5 on a large held-out part of train: either a global threshold, or a per-entity choice that maximizes expected F0.5. The stage (2 or 3) and the rule that score higher on the hold-out are used.
CPU only; no external data, APIs or pretrained models.

## Layout

```
src/
  config.py         paths (override with ER_DATA_DIR / ER_WORK_DIR / ER_OUTPUT_DIR)
  ingest.py         raw TSV -> parquet; ground truth exploded to (s1, other) pairs
  normalize.py      name / address normalization (transliteration, abbreviations, states)
  translit.py       learns an Indic-script -> Latin token dictionary from training matches
  prep.py           normalizes every source once
  blocking.py       per-country TF-IDF top-K retrieval (4 channels + exact name key)
  stage1.py         candidate pruner (LightGBM on retrieval signals) used inside blocking
  eval_blocking.py  blocking recall ceiling vs training ground truth
  features.py       pair features (rapidfuzz, token overlap, house numbers, context)
  match.py          stage-2 model: training (S1-entity folds), calibration, prediction, outputs
  cluster.py        stage 3: sibling (cluster-consistency) features + re-scoring model
  metrics.py        macro F0.5 exactly as the leaderboard defines it
  pipeline.py       runs every step in order
  eda.py            exploratory statistics
```

## Reproduce end to end

Expected data layout (the unzipped `student_resource/` next to this folder's parent):

```
<root>/student_resource/dataset/{train,test}/*.tsv
<root>/student_resource/utils/validate_submission.py
<root>/business_entity_resolution/src/...
```

```bash
pip install -r requirements.txt          # or requirements-min.txt on older Python
cd src
python pipeline.py                       # all steps -> <root>/output/*.tsv, then validator
```

Smoke test on a 2% sample of Source 2/3 (about 15 min on a laptop):

```bash
python pipeline.py --pct 2
```

Smaller machine (e.g. Colab High-RAM): train on 30% of train Source 2/3, predict on all of test.
When the train and test percentages differ, the S1-context features and the per-entity
decision rule are switched off, because they depend on how many S2/S3 records were blocked.

```bash
python pipeline.py --train-pct 30
```

On a 16 GB laptop, the full-density configuration (v7: all train records blocked and featurized, models fitted on 30 % of them, the other 70 % a hold-out) runs in about 4 hours. Memory stays bounded as follows:
- Every step and every feature / sibling part runs in its own process.
- Blocking streams the queries in chunks and spills results to disk.
- Every step resumes from its finished parts.

`tools/overnight.py` (repo root) runs the steps one by one. Each step goes through `tools/run_guarded.py`, which lowers CPU priority and stops the job if available RAM drops below the limit. The driver waits for free RAM and retries with smaller chunks:

```bash
ER_BLOCK_CHUNK=50000 python ../../tools/overnight.py --name v7 --train-pct 100 --test-pct 100 \
    --fit-pct 30 --steps ingest translit prep_train prep_test stage1_data stage1 block_train \
    block_test eval_blocking match_features_train match_features_test train score stage3 \
    calibrate predict validate
```

Resume from any step, e.g. after changing the model only:

```bash
python pipeline.py --steps train predict validate
```

France appears only in test (train is US + India), so predictions there get two guards (see
`match.finalize`): matches need `p >= threshold + ER_FR_DELTA` (default 0.1), and every S1 entity
keeps at most `ER_MAX_PER_S1` (default 11, the training maximum) records. To try other values
without re-scoring (seconds, rewrites `output/`):

```bash
python match.py decide --fr-delta 0.2      # also --max-per-s1 N
```

Environment knobs:
- `ER_THREADS` (default 16).
- `ER_BLOCK_CHUNK`: S2/S3 records per blocking chunk (default 100000).
- `ER_FEATURE_PART`: candidate pairs per feature part (default 200000).
- `ER_SIB_PART`: pairs per stage-3 sibling part (default 400000).
- `ER_PRUNE_TOP`, `ER_PRUNE_PMIN`, `ER_FOLDS`, `ER_ROUNDS`, `ER_ROUNDS3`.
- `ER_STAGE3=0` disables stage 3.

Lower the chunk and part sizes if memory is tight; they do not change the results.

## Resources (full data)

Planned for an instance with at least 16 vCPU and 64 GB RAM (e.g. SageMaker `ml.m5.4xlarge`,
or `ml.r5.4xlarge` for more memory headroom). Blocking dominates the run time; it scales
linearly with the number of Source 2/3 records and with `ER_THREADS`.

## Outputs

- `output/matching_results.tsv` — one row per test Source 1 entity, matched S2/S3 ids.
- `output/candidate_pairs.tsv` — the pruned candidate set the matching model scored.
