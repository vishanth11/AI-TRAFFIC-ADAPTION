"""Phase 1 CSV audit — DTC-FM feature/label datasets (read-only).

Profiles both CSVs, verifies feature<->label alignment, checks label
semantics evidence, and flags engineered (FE_*) / leakage-suspect columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path("dataset")
REPORT_DIR = Path("reports")
FEATURES_CSV = DATASET_DIR / "filtered_features_DTC_FM.csv"
LABELS_CSV = DATASET_DIR / "target_labels_DTC_FM.csv"


def profile_table(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\n-- dtypes --")
    print(df.dtypes.to_string())

    print("\n-- missing values --")
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    print(pd.DataFrame({"missing": miss, "missing_pct": miss_pct}).to_string())

    print("\n-- duplicates --")
    print(f"fully duplicated rows: {df.duplicated().sum()}")

    num = df.select_dtypes(include=[np.number])
    print("\n-- infinite values (numeric cols) --")
    inf_counts = np.isinf(num).sum()
    print(inf_counts[inf_counts > 0].to_string() if inf_counts.sum() else "none")

    print("\n-- constant columns --")
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(const if const else "none")

    print("\n-- numeric summary --")
    with pd.option_context("display.width", 200):
        print(num.describe().T.round(4).to_string())

    print("\n-- categorical cardinality --")
    for c in df.columns:
        if df[c].dtype == object or df[c].nunique() <= 20:
            vc = df[c].value_counts(dropna=False)
            if df[c].nunique() <= 20:
                print(f"\n{c}: {df[c].nunique()} unique")
                print((vc.to_frame("count").assign(pct=(vc / len(df) * 100).round(2))).to_string())


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    feats = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)

    profile_table(feats, "filtered_features_DTC_FM.csv")
    profile_table(labels, "target_labels_DTC_FM.csv")

    print(f"\n{'=' * 70}\nFEATURE/LABEL RELATIONSHIP\n{'=' * 70}")
    print(f"feature rows : {len(feats)}")
    print(f"label rows   : {len(labels)}")
    print(f"row counts match: {len(feats) == len(labels)}")
    print(f"labels has ID column: {list(labels.columns)}")

    lab = labels.iloc[:, 0]
    print(f"\nlabel column '{labels.columns[0]}' unique values: {sorted(lab.unique())}")
    dist = lab.value_counts().sort_index()
    print("\nclass distribution:")
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt / len(lab) * 100:.2f}%)")

    # Correlation of each feature with the label (relationship check only,
    # not a merge — labels are aligned positionally if counts match).
    if len(feats) == len(lab):
        print("\n-- point-biserial (pearson) correlation of numeric features vs label --")
        num = feats.select_dtypes(include=[np.number])
        corr = num.corrwith(lab).sort_values(key=np.abs, ascending=False)
        print(corr.round(4).to_string())

        print("\n-- feature correlation matrix (top |r| pairs) --")
        cm_vals = num.corr().abs().to_numpy().copy()
        np.fill_diagonal(cm_vals, 0)
        cm = pd.DataFrame(cm_vals, index=num.columns, columns=num.columns)
        pairs = (
            cm.stack().sort_values(ascending=False)
            .drop_duplicates()
            .head(15)
        )
        print(pairs.round(3).to_string())

    # Engineered-feature reverse-engineering checks (evidence only).
    print(f"\n{'=' * 70}\nENGINEERED FEATURE VERIFICATION (FE_*)\n{'=' * 70}")
    if "FE_inv_PET" in feats and "PET" in feats:
        expected = 1.0 / feats["PET"].replace(0, np.nan)
        err = (feats["FE_inv_PET"] - expected).abs().max()
        print(f"FE_inv_PET == 1/PET          : max abs err = {err:.2e}")

    if "FE_dist_squared" in feats and "target_dist_px" in feats:
        expected = feats["target_dist_px"] ** 2
        err = (feats["FE_dist_squared"] - expected).abs().max()
        print(f"FE_dist_squared == dist_px^2 : max abs err = {err:.2e}")

    if "FE_log_mesafe" in feats and "target_dist_px" in feats:
        for fn, name in [(np.log, "ln"), (np.log10, "log10"), (np.log1p, "ln(1+x)")]:
            expected = fn(feats["target_dist_px"].clip(lower=0))
            err = (feats["FE_log_mesafe"] - expected).abs().max()
            print(f"FE_log_mesafe == {name}(dist_px) : max abs err = {err:.2e}")

    if "FE_safety_index" in feats:
        print("FE_safety_index: no obvious single-source formula found by inspection;"
              " correlations with candidates below.")
        for c in ["FE_inv_PET", "PET", "min_dist_dual_check_m", "angle_degrees",
                  "target_dist_px", "FE_log_mesafe"]:
            if c in feats:
                r = feats["FE_safety_index"].corr(feats[c])
                print(f"  corr(FE_safety_index, {c}) = {r:+.4f}")
        # common composite guesses
        with np.errstate(divide="ignore", invalid="ignore"):
            guess_a = feats["FE_inv_PET"] / feats["FE_log_mesafe"]
            guess_b = feats["FE_inv_PET"] / (feats["min_dist_dual_check_m"] + 1)
        print(f"  corr(FE_safety_index, inv_PET/log_mesafe) = "
              f"{feats['FE_safety_index'].corr(guess_a):+.4f}")
        print(f"  corr(FE_safety_index, inv_PET/(min_dist+1)) = "
              f"{feats['FE_safety_index'].corr(guess_b):+.4f}")

    print("\n-- class-conditional feature means (label semantics evidence) --")
    if len(feats) == len(lab):
        num = feats.select_dtypes(include=[np.number])
        with pd.option_context("display.width", 250):
            print(num.groupby(lab).mean().T.round(3).to_string())


if __name__ == "__main__":
    main()