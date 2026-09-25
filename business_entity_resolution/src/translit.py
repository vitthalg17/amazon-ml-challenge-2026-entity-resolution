"""Learn an Indic-script token -> Latin token dictionary from training matches.

Names: when a Source 2/3 name contains Indic tokens and has the same token count as
its matched Source 1 name, tokens are aligned by position.
Addresses (reordered, so no positional alignment): each Indic token maps to the
Source 1 address token that co-occurs with it most consistently (ties -> rarer token).
Keys are the anyascii form of the Indic token (see normalize.ascii_key).
"""
import argparse
import json
import re
from collections import Counter, defaultdict

import polars as pl
from anyascii import anyascii

from config import SEED, WORK_DIR, pq_path
from normalize import INDIC_RE, TRANSLIT_PATH, ascii_key

TOK_RE = re.compile(r"[^a-z0-9]+")


def latin_tokens(s: str) -> list[str]:
    return [t for t in TOK_RE.split(anyascii(s).lower()) if t]


def other_tokens(s: str) -> list[tuple[str, bool]]:
    out = []
    for raw in re.split(r"[\s,]+", s):
        if INDIC_RE.search(raw):
            k = ascii_key(raw)
            if k:
                out.append((k, True))
        else:
            out.extend((t, False) for t in latin_tokens(raw))
    return out


def load_pairs(pct: int) -> pl.DataFrame:
    pairs = pl.scan_parquet(WORK_DIR / "train_gt_pairs.parquet")
    others = pl.concat([pl.scan_parquet(pq_path("train", s)) for s in (2, 3)])
    others = others.filter(
        (pl.col("entity_id").hash(SEED) % 100 < pct) & (
        pl.col("business_name").str.contains(INDIC_RE.pattern)
        | pl.col("business_address").str.contains(INDIC_RE.pattern)))
    df = (others.join(pairs, left_on="entity_id", right_on="other")
          .join(pl.scan_parquet(pq_path("train", 1)), left_on="s1", right_on="entity_id",
                suffix="_s1")
          .select("business_name", "business_address", "business_name_s1",
                  "business_address_s1")
          .collect())
    return df


def learn(df: pl.DataFrame, min_count: int = 2, min_share: float = 0.5):
    name_cnt: dict[str, Counter] = defaultdict(Counter)
    addr_cnt: dict[str, Counter] = defaultdict(Counter)
    addr_tot: Counter = Counter()
    s1_df: Counter = Counter()

    for n_o, a_o, n_1, a_1 in df.iter_rows():
        if INDIC_RE.search(n_o):
            to, t1 = other_tokens(n_o), latin_tokens(n_1)
            if len(to) == len(t1):
                for (k, indic), u in zip(to, t1):
                    if indic:
                        name_cnt[k][u] += 1
        if a_o and INDIC_RE.search(a_o):
            s1_toks = set(latin_tokens(a_1))
            s1_df.update(s1_toks)
            for k, indic in set(other_tokens(a_o)):
                if indic:
                    addr_tot[k] += 1
                    addr_cnt[k].update(s1_toks)

    name_map = {}
    for k, c in name_cnt.items():
        u, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            name_map[k] = u

    addr_map = {}
    for k, c in addr_cnt.items():
        tot = addr_tot[k]
        best = max(c.values())
        if best < min_count or best / tot < min_share:
            continue
        cands = [u for u, n in c.items() if n >= 0.95 * best]
        addr_map[k] = min(cands, key=lambda u: s1_df[u])
    return name_map, addr_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=100,
                    help="percent of Source 2/3 records to learn from (hash sample)")
    args = ap.parse_args()
    df = load_pairs(args.pct)
    print("pairs with Indic text:", df.height)
    name_map, addr_map = learn(df)
    print("name entries:", len(name_map), "addr entries:", len(addr_map))
    with open(TRANSLIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name_map, "addr": addr_map}, f, ensure_ascii=False)
    for m in (name_map, addr_map):
        print(list(m.items())[:25])


if __name__ == "__main__":
    main()
