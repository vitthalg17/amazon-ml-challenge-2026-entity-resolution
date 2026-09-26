"""Pair features for the matching model.

Groups:
  retrieval  - the three blocking cosines and per-channel ranks
  name       - rapidfuzz similarities on core / full names, token Jaccard, DBA alt-name match
  address    - token Jaccard / containment, fuzzy ratios, house-number agreement / conflict
  place      - name without place words (name_loc) and street-name similarity, plus how many
               Source 1 entities share the name: when a name is common ("Bordeaux Club" x493)
               only the street can tell the businesses apart
  rarity     - IDF-weighted overlap (IDF from all Source 1 records): a shared rare token is strong
               evidence for a match, a rare token on one side only is strong evidence against
  record     - lengths, empty address, Indic script, domain-as-name, source
  context    - how this pair compares with the other candidates of the same S2/S3 record
               and of the same S1 entity (each S2/S3 record belongs to at most one S1)
"""
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

N_WORKERS = int(os.environ.get("ER_THREADS", os.cpu_count() or 4))
SIDE_COLS = ["entity_id", "business_name", "name_norm", "name_core", "name_alt",
             "addr_norm", "addr_nums", "name_loc", "street"]

FEATURES: list[str] = []  # filled by build_features (order used by the model)


def _cp(a: list[str], b: list[str], scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=N_WORKERS, dtype=np.float32)


def _jacc(a: str, b: str) -> list[pl.Expr]:
    # extract_all (not split) so "" -> [] and two empty fields don't score as a perfect match
    A, B = pl.col(a).str.extract_all(r"\S+"), pl.col(b).str.extract_all(r"\S+")
    inter = A.list.set_intersection(B).list.len()
    return [
        (inter / A.list.set_union(B).list.len().clip(1)).cast(pl.Float32),
        (inter / A.list.len().clip(1)).cast(pl.Float32),
        inter.cast(pl.Int16),
    ]


def token_idf(texts: pl.Series) -> tuple[pl.Series, pl.Series, float]:
    """(tokens, idf, idf of an unseen token) over space-separated documents."""
    toks = texts.str.extract_all(r"\S+").list.unique().explode().drop_nulls().alias("tok")
    vc = toks.value_counts(name="df")
    n = texts.len()
    idf = pl.Series(np.log((n + 1) / (vc["df"].to_numpy() + 1)) + 1, dtype=pl.Float32)
    return vc["tok"], idf, float(np.log(n + 1) + 1)


def _idf(a: str, b: str, idf) -> list[pl.Expr]:
    keys, vals, unseen = idf

    def w(lst: pl.Expr) -> pl.Expr:
        return lst.list.eval(pl.element().replace_strict(keys, vals, default=unseen,
                                                         return_dtype=pl.Float32))

    A, B = pl.col(a).str.extract_all(r"\S+"), pl.col(b).str.extract_all(r"\S+")
    inter = A.list.set_intersection(B)
    return [
        (w(inter).list.sum() / w(A.list.set_union(B)).list.sum().clip(1e-6)).cast(pl.Float32),
        w(inter).list.max().fill_null(0.0),                        # rarest shared token
        w(A.list.set_difference(B)).list.max().fill_null(0.0),     # rarest token only in S2/S3
        w(B.list.set_difference(A)).list.max().fill_null(0.0),     # rarest token only in S1
    ]


CTX_BASE = ("cos_name_word", "cos_name_char", "cos_addr", "cos_combo", "cos_sum", "p1",
            "n_tset", "a_tset", "nl_tset", "st_tset")


def build_features(cand: pl.DataFrame, s1: pl.DataFrame, oth: pl.DataFrame,
                   name_idf=None, addr_idf=None) -> pl.DataFrame:
    """cand: blocking output (other_id, s1_id, cos_*, rank_*). s1/oth: normalized sources.
    name_idf / addr_idf: token_idf() over all Source 1 names / addresses (default: over `s1`)."""
    df = pair_features(cand, s1, oth, name_idf, addr_idf)
    df = with_keys(df).join(context_features(context_base(df)), on=KEYS).drop(KEYS)
    FEATURES[:] = [c for c in df.columns if c not in ("other_id", "s1_id", "country", "label")]
    return df


KEYS = ["_o", "_s"]  # 64-bit hashes of other_id / s1_id: cheap group keys for the context step


def with_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_o=pl.col("other_id").hash(), _s=pl.col("s1_id").hash())


def context_base(df: pl.DataFrame) -> pl.DataFrame:
    """The slim table context_features needs: hashed ids + CTX_BASE (~48 bytes per pair)."""
    return df.select(_o=pl.col("other_id").hash(), _s=pl.col("s1_id").hash(), *CTX_BASE)


def context_features(base: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Per-pair comparison with the other candidates of the same S2/S3 record (_o) and S1
    entity (_s). Needs every candidate of those groups at once, but only the slim context_base
    table, so it runs over all candidate pairs even when pair features were built in chunks.
    Returns KEYS + context columns."""
    df = base.lazy()
    ctx = []
    for c in CTX_BASE:
        ctx += [
            (pl.col(c).max().over("_o") - pl.col(c)).alias(f"{c}_gap_o"),
            (pl.col(c).max().over("_s") - pl.col(c)).alias(f"{c}_gap_s"),
        ]
    ctx += [
        pl.col("cos_sum").rank("ordinal", descending=True).over("_o").cast(pl.Int16).alias("rank_o"),
        pl.col("cos_sum").rank("ordinal", descending=True).over("_s").cast(pl.Int16).alias("rank_s"),
        pl.len().over("_o").cast(pl.Int16).alias("ncand_o"),
        pl.len().over("_s").cast(pl.Int16).alias("ncand_s"),
        # second-best competitor margin for the S2/S3 record
        (pl.col("cos_sum") - pl.col("cos_sum").sort(descending=True).slice(1, 1).first()
         .over("_o")).fill_null(1.0).alias("cos_sum_margin2_o"),
    ]
    passthrough = [c for c in ("_p",) if c in df.collect_schema().names()]  # part id, if any
    return df.select(*KEYS, *passthrough, *ctx).collect()


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, oth: pl.DataFrame,
                  name_idf=None, addr_idf=None) -> pl.DataFrame:
    """All features that depend on one (S2/S3, S1) pair only; safe to compute in chunks.
    s1 should be all of Source 1: name frequencies are counted over it."""
    name_idf = name_idf or token_idf(s1["name_core"])
    addr_idf = addr_idf or token_idf(s1["addr_norm"])
    freq = s1.group_by("name_core").agg(pl.len().cast(pl.Int32).alias("_freq"))
    df = (cand
          .join(oth.select(SIDE_COLS), left_on="other_id", right_on="entity_id")
          .join(s1.select(SIDE_COLS), left_on="s1_id", right_on="entity_id", suffix="_1")
          .join(freq.rename({"_freq": "n_core_freq_o"}), on="name_core", how="left")
          .join(freq.rename({"name_core": "name_core_1", "_freq": "n_core_freq_1"}),
                on="name_core_1", how="left")
          .with_columns(pl.col("n_core_freq_o").fill_null(0)))

    nc, nc1 = df["name_core"].to_list(), df["name_core_1"].to_list()
    nn, nn1 = df["name_norm"].to_list(), df["name_norm_1"].to_list()
    ad, ad1 = df["addr_norm"].to_list(), df["addr_norm_1"].to_list()
    nl, nl1 = df["name_loc"].to_list(), df["name_loc_1"].to_list()
    st, st1 = df["street"].to_list(), df["street_1"].to_list()
    fz = {
        "n_ratio": _cp(nc, nc1, fuzz.ratio),
        "n_tsort": _cp(nc, nc1, fuzz.token_sort_ratio),
        "n_tset": _cp(nc, nc1, fuzz.token_set_ratio),
        "n_partial": _cp(nc, nc1, fuzz.partial_ratio),
        "n_jw": _cp(nc, nc1, JaroWinkler.normalized_similarity),
        "n_full_tsort": _cp(nn, nn1, fuzz.token_sort_ratio),
        "n_alt_tset": _cp(df["name_alt"].to_list(), nc1, fuzz.token_set_ratio),
        "a_tset": _cp(ad, ad1, fuzz.token_set_ratio),
        "a_tsort": _cp(ad, ad1, fuzz.token_sort_ratio),
        "a_partial": _cp(ad, ad1, fuzz.partial_ratio),
        "nl_tset": _cp(nl, nl1, fuzz.token_set_ratio),
        "nl_ratio": _cp(nl, nl1, fuzz.ratio),
        "st_tset": _cp(st, st1, fuzz.token_set_ratio),
        "st_ratio": _cp(st, st1, fuzz.ratio),
        "st_partial": _cp(st, st1, fuzz.partial_ratio),
    }
    df = df.with_columns(**{k: pl.Series(v) for k, v in fz.items()})
    del nc, nc1, nn, nn1, ad, ad1, nl, nl1, st, st1, fz
    # place features are missing (not 0) when a side has no such tokens
    nl_both = (pl.col("name_loc") != "") & (pl.col("name_loc_1") != "")
    st_both = (pl.col("street") != "") & (pl.col("street_1") != "")
    df = df.with_columns(
        *[pl.when(nl_both).then(pl.col(c)).alias(c) for c in ("nl_tset", "nl_ratio")],
        *[pl.when(st_both).then(pl.col(c)).alias(c) for c in ("st_tset", "st_ratio", "st_partial")],
        nl_both=nl_both.cast(pl.Int8), st_both=st_both.cast(pl.Int8),
    )

    nj, nc_, ni = _jacc("name_core", "name_core_1")
    aj, ac, ai = _jacc("addr_norm", "addr_norm_1")
    mj, mc, mi = _jacc("addr_nums", "addr_nums_1")
    lj, lc, _ = _jacc("name_loc", "name_loc_1")
    sj, sc, _ = _jacc("street", "street_1")
    nw = dict(zip(("n_idf_jacc", "n_idf_shared_max", "n_idf_only_o", "n_idf_only_s"),
                  _idf("name_core", "name_core_1", name_idf)))
    aw = dict(zip(("a_idf_jacc", "a_idf_shared_max", "a_idf_only_o", "a_idf_only_s"),
                  _idf("addr_norm", "addr_norm_1", addr_idf)))
    first = pl.col("addr_nums").str.extract(r"^(\d+)")
    first1 = pl.col("addr_nums_1").str.extract(r"^(\d+)")
    raw = pl.col("business_name")
    df = df.with_columns(
        n_jacc=nj, n_contain=nc_, n_inter=ni,
        a_jacc=aj, a_contain=ac, a_inter=ai,
        num_jacc=mj, num_contain=mc, num_inter=mi,
        nl_jacc=pl.when(nl_both).then(lj), nl_contain=pl.when(nl_both).then(lc),
        st_jacc=pl.when(st_both).then(sj), st_contain=pl.when(st_both).then(sc),
        # name tokens that are place words (city / country) on each side
        n_loc_ntok=(pl.col("name_core").str.count_matches(r"\S+")
                    - pl.col("name_loc").str.count_matches(r"\S+")).cast(pl.Int16),
        n_loc_ntok_1=(pl.col("name_core_1").str.count_matches(r"\S+")
                      - pl.col("name_loc_1").str.count_matches(r"\S+")).cast(pl.Int16),
        # both addresses carry numbers and none agree (e.g. different house / plot numbers)
        num_conflict=((pl.col("addr_nums") != "") & (pl.col("addr_nums_1") != "")
                      & (mi == 0)).cast(pl.Int8),
        **nw, **aw,
        num_first_eq=(first == first1).fill_null(False).cast(pl.Int8),
        num_first_prefix=((first.is_not_null() & first1.is_not_null())
                          & (first1.str.starts_with(first.fill_null("#"))
                             | first.str.starts_with(first1.fill_null("#")))).fill_null(False).cast(pl.Int8),
        n_first_tok_eq=(pl.col("name_core").str.extract(r"^(\S+)")
                        == pl.col("name_core_1").str.extract(r"^(\S+)")).fill_null(False).cast(pl.Int8),
        n_len=pl.col("name_core").str.len_chars().cast(pl.Int16),
        n_len_1=pl.col("name_core_1").str.len_chars().cast(pl.Int16),
        n_ntok=pl.col("name_core").str.count_matches(r"\S+").cast(pl.Int16),
        n_ntok_1=pl.col("name_core_1").str.count_matches(r"\S+").cast(pl.Int16),
        a_ntok=pl.col("addr_norm").str.count_matches(r"\S+").cast(pl.Int16),
        a_ntok_1=pl.col("addr_norm_1").str.count_matches(r"\S+").cast(pl.Int16),
        a_nnum=pl.col("addr_nums").str.count_matches(r"\S+").cast(pl.Int16),
        addr_empty=(pl.col("addr_norm") == "").cast(pl.Int8),
        name_indic=raw.str.contains(r"[ऀ-෿]").cast(pl.Int8),
        name_domain=raw.str.contains(r"(?i)\.(com|net|org|in|co|biz|info|fr|io)\b|www\.").cast(pl.Int8),
        name_upper=(raw == raw.str.to_uppercase()).cast(pl.Int8),
        has_alt=(pl.col("name_alt") != "").cast(pl.Int8),
        src3=pl.col("other_id").str.starts_with("S3-").cast(pl.Int8),
    )

    df = df.with_columns(cos_sum=pl.sum_horizontal(pl.col("^cos_.*$")))
    drop = [c for c in df.columns if c in SIDE_COLS or c.endswith("_1") and c[:-2] in SIDE_COLS]
    return df.drop([c for c in drop if c != "entity_id"], strict=False)
