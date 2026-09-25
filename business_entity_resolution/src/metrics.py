"""Macro F0.5 exactly as the leaderboard defines it (per Source 1 entity, singletons included)."""
import polars as pl


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred / gt: pair frames with columns (s1, other). s1_ids: the Source 1 entities scored.

    Per entity: F0.5 = 1.25*tp / (0.25*n_true + n_pred); 1.0 when both lists are empty.
    """
    ids = pl.DataFrame({"s1": s1_ids.unique()})
    pred = pred.select("s1", "other").join(ids, on="s1", how="semi")
    gt = gt.select("s1", "other").join(ids, on="s1", how="semi")
    tp = pred.join(gt, on=["s1", "other"], how="inner").group_by("s1").agg(tp=pl.len())
    per = (ids
           .join(gt.group_by("s1").agg(n_true=pl.len()), on="s1", how="left")
           .join(pred.group_by("s1").agg(n_pred=pl.len()), on="s1", how="left")
           .join(tp, on="s1", how="left")
           .fill_null(0)
           .with_columns(f=pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
                         .otherwise(1.25 * pl.col("tp")
                                    / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9))))
    single = per.filter(pl.col("n_true") == 0)
    multi = per.filter(pl.col("n_true") > 0)
    return {
        "macro_f05": per["f"].mean(),
        "singleton_acc": single["f"].mean() if single.height else float("nan"),
        "nonsingleton_f05": multi["f"].mean() if multi.height else float("nan"),
        "pair_precision": tp["tp"].sum() / max(pred.height, 1),
        "pair_recall": tp["tp"].sum() / max(gt.height, 1),
        "n_entities": per.height,
        "n_singletons": single.height,
    }
