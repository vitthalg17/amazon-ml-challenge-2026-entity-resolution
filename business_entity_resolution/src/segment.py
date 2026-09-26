"""Split run-together business names into words: "northbrothers" -> "north brothers",
"infotechgood.com" -> "infotech good", "indiaparasengineering" -> "india paras engineering".

Some Source 2/3 records write the name as one token (web domains, social handles). Word-based
retrieval and similarities cannot match those against the spaced Source 1 name. The vocabulary
is the word counts of the Source 1 names (train and test; names only, no labels). Every long
token that is rare in that vocabulary is split, when possible, into the most probable sequence
of common words (unigram model, dynamic programming). The table token -> words is applied to
every source by normalize.name_exprs, so identical names stay identical.
Output: <work>/name_splits.json
"""
import json
import math
import time

import polars as pl

from config import WORK_DIR, pq_path

SPLITS_PATH = WORK_DIR / "name_splits.json"
MIN_LEN = 7          # only tokens at least this long are split
MAX_KNOWN = 2        # tokens seen this often among S1 names are left alone
MIN_PIECE = 3        # shortest word a token is split into
MIN_PIECE_COUNT = 5  # words must be this common among S1 names
MAX_PIECES = 4


def _tokens(split: str, source: int, min_len: int = 1) -> pl.LazyFrame:
    """Name tokens as normalize.name_exprs sees them before the split table (accents folded;
    other scripts are dropped here - their transliterations are never split)."""
    import normalize
    name = (pl.col("business_name").fill_null("").str.normalize("NFKD")
            .str.replace_all(r"\p{M}", "").str.to_lowercase())
    return (pl.scan_parquet(pq_path(split, source))
            .select(tok=normalize.clean_name(name).str.split(" "))
            .explode("tok").filter(pl.col("tok").str.len_chars() >= min_len))


def split_token(t: str, count: dict, n_total: int) -> str | None:
    """Most probable split of t into >= 2 known words (None if there is none)."""
    n = len(t)
    best = [None] * (n + 1)  # best[i]: (log-prob, pieces) of the prefix t[:i]
    best[0] = (0.0, [])
    for i in range(MIN_PIECE, n + 1):
        for j in range(max(0, i - 25), i - MIN_PIECE + 1):
            if best[j] is None or len(best[j][1]) >= MAX_PIECES:
                continue
            c = count.get(t[j:i], 0)
            if c < MIN_PIECE_COUNT:
                continue
            sc = best[j][0] + math.log(c / n_total)
            if best[i] is None or sc > best[i][0]:
                best[i] = (sc, best[j][1] + [t[j:i]])
    r = best[n]
    return " ".join(r[1]) if r is not None and len(r[1]) >= 2 else None


def build() -> dict:
    t0 = time.time()
    vocab = (pl.concat([_tokens(sp, 1) for sp in ("train", "test")])
             .group_by("tok").agg(n=pl.len()).collect(engine="streaming"))
    count = dict(zip(vocab["tok"].to_list(), vocab["n"].to_list()))
    n_total = int(vocab["n"].sum())
    cands = (pl.concat([_tokens(sp, s, MIN_LEN) for sp in ("train", "test") for s in (1, 2, 3)])
             .unique().collect(engine="streaming")["tok"].to_list())
    cands = [t for t in cands if count.get(t, 0) < MAX_KNOWN and not t.isdigit()]
    splits = {}
    for t in cands:
        r = split_token(t, count, n_total)
        if r is not None:
            splits[t] = r
    with open(SPLITS_PATH, "w", encoding="utf-8") as fh:
        json.dump(splits, fh)
    print(f"segment: {len(vocab):,} S1 words, {len(cands):,} rare long tokens, "
          f"{len(splits):,} split ({time.time() - t0:.0f}s); e.g. "
          + ", ".join(f"{k} -> {v}" for k, v in list(splits.items())[:8]), flush=True)
    return splits


if __name__ == "__main__":
    build()
