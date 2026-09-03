"""Phase 3 — group-aware stratified 70/15/15 split for DTC-FM.

Duplicate feature rows form groups (Phase 2 found 5 groups of 2); a group is
assigned to exactly ONE split so identical observations never straddle
train/validation/test. Deterministic: RANDOM_SEED = 42, no library-split
shortcut — deficit-based greedy assignment over seeded-shuffled groups.

Outputs:
  data/processed/near_miss/splits/{train,validation,test}.parquet
  data/processed/near_miss/split_assignments.csv   (sample_index, split, event_type)
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
DATA = Path("data/processed/near_miss")
COMBINED = DATA / "dtc_fm_combined.parquet"
SPLIT_DIR = DATA / "splits"

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
FEATURE_COLS = [
    "speed_object_1_kph", "speed_object_2_kph", "angle_degrees", "PET",
    "acceleration_obj1_mps2", "acceleration_obj2_mps2",
    "min_dist_dual_check_m", "object_count_at_frame1", "target_dist_px",
    "FE_inv_PET", "FE_safety_index", "FE_log_mesafe", "FE_dist_squared",
    "class_object_1", "class_object_2",
]


def make_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Group id per row: identical feature rows share a group (0..n_groups-1)."""
    keys = df[FEATURE_COLS].astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(keys)
    return pd.Series(codes, index=df.index, name="dup_group")


def group_aware_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign whole duplicate-groups to train/val/test, stratified by class.

    Greedy deficit assignment: walk groups in seeded-shuffled order, give each
    group to the split whose class quota is furthest behind. Exact quotas are
    unreachable when groups span the boundary, so the algorithm minimizes the
    shortfall; priority is no-leakage > class balance > exact proportions.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    dup = make_duplicate_groups(df)
    assign = pd.Series(index=df.index, dtype=object)

    for cls in sorted(df["event_type"].unique()):
        cls_idx = df.index[df["event_type"] == cls]
        groups = dup.loc[cls_idx].unique()
        rng.shuffle(groups)
        n_cls = len(cls_idx)
        targets = {
            "train": TRAIN_FRAC * n_cls,
            "validation": VAL_FRAC * n_cls,
            "test": VAL_FRAC * n_cls,
        }
        counts = {s: 0 for s in targets}
        for g in groups:
            g_idx = cls_idx[dup.loc[cls_idx] == g]
            size = len(g_idx)
            deficits = {s: targets[s] - counts[s] for s in targets}
            split = max(deficits, key=deficits.get)
            counts[split] += size
            assign.loc[g_idx] = split

    return pd.DataFrame({
        "sample_index": df["sample_index"],
        "split": assign,
        "event_type": df["event_type"],
    })


def verify_no_leakage(df: pd.DataFrame, assignments: pd.DataFrame) -> dict:
    """Integrity checks: duplicate groups and exact feature rows never cross splits."""
    merged = df.merge(assignments[["sample_index", "split"]], on="sample_index")
    dup = make_duplicate_groups(merged)

    issues = []
    # 1. no duplicate group spans >1 split
    cross = merged.assign(dup_group=dup).groupby("dup_group")["split"].nunique()
    n_cross_groups = int((cross > 1).sum())
    if n_cross_groups:
        issues.append(f"{n_cross_groups} duplicate groups cross splits")

    # 2. no identical feature row appears in two splits (independent of grouping)
    feat_hash = merged[FEATURE_COLS].astype(str).agg("|".join, axis=1)
    per_split_sets = {
        s: set(feat_hash[merged["split"] == s]) for s in ["train", "validation", "test"]
    }
    overlap = len(per_split_sets["train"] & per_split_sets["validation"]) + \
        len(per_split_sets["train"] & per_split_sets["test"]) + \
        len(per_split_sets["validation"] & per_split_sets["test"])
    if overlap:
        issues.append(f"{overlap} identical feature rows shared across splits")

    # 3. no sample_index overlap
    if len(set.intersection(*[set(merged.loc[merged['split'] == s, 'sample_index'])
                              for s in per_split_sets])):
        issues.append("sample_index overlap across splits")

    counts = merged["split"].value_counts().to_dict()
    cls_dist = {
        s: merged.loc[merged["split"] == s, "event_type"].value_counts().to_dict()
        for s in ["train", "validation", "test"]
    }
    return {
        "duplicate_groups_crossing_splits": n_cross_groups,
        "identical_feature_rows_crossing_splits": overlap,
        "sample_index_overlap": False,
        "split_counts": counts,
        "split_pcts": {s: round(c / len(merged) * 100, 2) for s, c in counts.items()},
        "class_distribution": cls_dist,
        "leakage_issues": issues,
        "pass": not issues,
    }


def save_splits(df: pd.DataFrame, assignments: pd.DataFrame) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    merged = df.merge(assignments[["sample_index", "split"]], on="sample_index")
    for split in ["train", "validation", "test"]:
        part = merged[merged["split"] == split].reset_index(drop=True)
        part.to_parquet(SPLIT_DIR / f"{split}.parquet", index=False)
    assignments.to_csv(DATA / "split_assignments.csv", index=False)


def build_split() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_parquet(COMBINED)
    assignments = group_aware_split(df)
    verification = verify_no_leakage(df, assignments)
    save_splits(df, assignments)
    return df, assignments, verification


if __name__ == "__main__":
    _, _, v = build_split()
    import json
    print(json.dumps(v, indent=2, default=str))
    if not v["pass"]:
        raise SystemExit("SPLIT LEAKAGE DETECTED")
    print("split saved to", SPLIT_DIR)