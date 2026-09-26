# Business Entity Resolution — Amazon ML Challenge 2026

Two-stage pipeline: **blocking** (multi-channel TF-IDF retrieval + a stage-1 pruner) followed by
a **LightGBM matching model** with a one-S1-per-record assignment and an F0.5-tuned decision
rule (a global threshold, or a per-entity choice that maximizes expected F0.5, whichever scores
higher out of fold).
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
  match.py          stage-2 model: training (S1-entity folds), threshold, prediction, outputs
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

On a 16 GB laptop (the configuration that produced leaderboard 0.9545): every step and every
feature part runs in its own process, and `tools/run_guarded.py` (repo root) adds low CPU
priority plus a watchdog that stops the job if available RAM drops below the limit:

```bash
ER_THREADS=16 ER_STAGE1_PCT=2 python ../../tools/run_guarded.py --min-avail-mb 2000 \
    --log run.log -- python -u pipeline.py --train-pct 30
```

Resume from any step, e.g. after changing the model only:

```bash
python pipeline.py --steps train predict validate
```

Environment knobs: `ER_THREADS` (default: all cores), `ER_BLOCK_CHUNK` (S2/S3 records per
blocking chunk, default 300000, lower it if memory is tight), `ER_PRUNE_TOP`, `ER_PRUNE_PMIN`,
`ER_FOLDS`, `ER_ROUNDS`.

## Resources (full data)

Planned for an instance with at least 16 vCPU and 64 GB RAM (e.g. SageMaker `ml.m5.4xlarge`,
or `ml.r5.4xlarge` for more memory headroom). Blocking dominates the run time; it scales
linearly with the number of Source 2/3 records and with `ER_THREADS`.

## Outputs

- `output/matching_results.tsv` — one row per test Source 1 entity, matched S2/S3 ids.
- `output/candidate_pairs.tsv` — the pruned candidate set the matching model scored.
