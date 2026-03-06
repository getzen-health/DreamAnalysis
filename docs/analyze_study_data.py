#!/usr/bin/env python3
"""
Neural Dream Workshop — Pilot Study Data Analysis
===================================================

Reads the CSV export from /api/study/admin/export-csv and computes all
statistics needed for the research paper.

Usage:
    python analyze_study_data.py study-data-2026-03-15.csv

Output:
    - Console tables (copy-paste into paper)
    - figures/ folder with publication-ready plots
    - results.json with all computed statistics
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Optional: plotting (graceful degradation if not installed)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("[WARN] matplotlib/seaborn not installed — skipping plots")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ──────────────────────────────────────────────────────────────────

EEG_BANDS = ["alpha", "beta", "theta", "delta", "gamma"]
MIN_QUALITY = 30        # Minimum quality score to include
MIN_SAMPLES = 10        # Not directly in CSV, but quality score proxies this
ALPHA_LEVEL = 0.05      # Significance level
BONFERRONI_TESTS_RQ1 = 6  # Number of comparisons for RQ1
BONFERRONI_TESTS_RQ2 = 4  # Number of comparisons for RQ2


# ── Helpers ────────────────────────────────────────────────────────────────────

def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Paired Cohen's d (within-subjects)."""
    diff = x - y
    return float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0


def safe_corr(x: pd.Series, y: pd.Series) -> tuple:
    """Spearman correlation, returns (rho, p). Handles NaN."""
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return (np.nan, np.nan)
    return stats.spearmanr(x[mask], y[mask])


def fmt(v, decimals=3):
    """Format a number for paper tables."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def fmt_p(p):
    """Format p-value for paper."""
    if p is None or np.isnan(p):
        return "—"
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"


def stars(p):
    """Significance stars."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Load and clean data ───────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV export and parse columns."""
    df = pd.read_csv(csv_path)

    # Normalize column names (the CSV uses snake_case)
    df.columns = df.columns.str.strip()

    # Parse numeric columns
    for band in EEG_BANDS:
        for prefix in ["pre_", "post_"]:
            col = f"{prefix}{band}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["age", "data_quality_score", "duration_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse boolean
    if "intervention_triggered" in df.columns:
        df["intervention_triggered"] = df["intervention_triggered"].astype(str).str.lower().isin(["true", "1"])
    if "has_apple_watch" in df.columns:
        df["has_apple_watch"] = df["has_apple_watch"].astype(str).str.lower().isin(["true", "1"])

    # Compute derived features
    for prefix in ["pre_", "post_"]:
        a = df[f"{prefix}alpha"]
        b = df[f"{prefix}beta"]
        t = df[f"{prefix}theta"]
        df[f"{prefix}alpha_beta_ratio"] = a / b.replace(0, np.nan)
        df[f"{prefix}theta_beta_ratio"] = t / b.replace(0, np.nan)

    return df


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to complete, quality sessions."""
    before = len(df)
    # Keep only non-partial (partial column may not exist if all complete)
    if "partial" in df.columns:
        df = df[df["partial"].astype(str).str.lower() != "true"]

    # Quality filter
    if "data_quality_score" in df.columns:
        df = df[df["data_quality_score"] >= MIN_QUALITY]

    after = len(df)
    print(f"[DATA] {before} rows → {after} after quality filter (min score={MIN_QUALITY})")
    return df


# ── Section 3.1: Demographics ─────────────────────────────────────────────────

def demographics(df: pd.DataFrame) -> dict:
    """Table 1: Participant demographics."""
    # Unique participants
    ppl = df.drop_duplicates(subset="participant_code")
    n = len(ppl)

    # Both sessions
    sess_counts = df.groupby("participant_code")["block_type"].nunique()
    both = int((sess_counts == 2).sum())

    age_m = ppl["age"].mean()
    age_sd = ppl["age"].std()

    diet_counts = ppl["diet_type"].value_counts()
    aw = int(ppl["has_apple_watch"].sum()) if "has_apple_watch" in ppl.columns else 0

    print("\n" + "=" * 60)
    print("TABLE 1: Participant Demographics")
    print("=" * 60)
    print(f"  Total enrolled:          {n}")
    print(f"  Both sessions complete:  {both}")
    print(f"  Age, M (SD):             {fmt(age_m, 1)} ({fmt(age_sd, 1)})")
    for diet in ["omnivore", "vegetarian", "vegan", "other"]:
        c = int(diet_counts.get(diet, 0))
        pct = c / n * 100 if n > 0 else 0
        print(f"  Diet: {diet:<15s}    {c} ({pct:.0f}%)")
    print(f"  Apple Watch owner:       {aw} ({aw/n*100:.0f}%)" if n > 0 else "  Apple Watch owner: 0")

    return {
        "n_participants": n,
        "n_both_complete": both,
        "age_mean": age_m,
        "age_sd": age_sd,
        "diet_counts": diet_counts.to_dict(),
        "apple_watch_count": aw,
    }


# ── Section 3.2: Data Quality ─────────────────────────────────────────────────

def data_quality(df: pd.DataFrame) -> dict:
    """Table 2: Data quality metrics per block type."""
    print("\n" + "=" * 60)
    print("TABLE 2: Data Quality")
    print("=" * 60)

    result = {}
    for block in ["stress", "food"]:
        sub = df[df["block_type"] == block]
        n_started = len(sub)
        # For completion rate, we'd need partial info; assume all in filtered set are complete
        n_complete = len(sub)
        q_m = sub["data_quality_score"].mean()
        q_sd = sub["data_quality_score"].std()
        d_m = sub["duration_seconds"].mean()
        d_sd = sub["duration_seconds"].std()

        intv = sub["intervention_triggered"].sum() if block == "stress" else "N/A"

        print(f"\n  {block.upper()} SESSION:")
        print(f"    Sessions completed:      {n_complete}")
        print(f"    Mean quality score (SD):  {fmt(q_m, 1)} ({fmt(q_sd, 1)})")
        print(f"    Mean duration, sec (SD):  {fmt(d_m, 0)} ({fmt(d_sd, 0)})")
        if block == "stress":
            print(f"    Intervention triggered:   {intv} ({intv/n_complete*100:.0f}%)" if n_complete > 0 else "")

        result[block] = {
            "n_complete": n_complete,
            "quality_mean": q_m,
            "quality_sd": q_sd,
            "duration_mean": d_m,
            "duration_sd": d_sd,
        }
        if block == "stress":
            result[block]["intervention_triggered"] = int(intv) if isinstance(intv, (int, np.integer)) else 0

    return result


# ── Section 3.3: RQ1 — Stress and EEG ─────────────────────────────────────────

def rq1_stress(df: pd.DataFrame) -> dict:
    """RQ1: Baseline vs post-session band powers for stress sessions."""
    stress = df[df["block_type"] == "stress"].copy()
    n = len(stress)

    print("\n" + "=" * 60)
    print(f"TABLE 3: RQ1 — Stress Session EEG (n={n})")
    print("Baseline vs. Post-Session Comparison")
    print("=" * 60)

    if n < 3:
        print("  [SKIP] Too few stress sessions for analysis")
        return {"n": n, "skip": True}

    results = {}
    bands_plus = EEG_BANDS + ["alpha_beta_ratio"]

    header = f"{'Band':<20s} {'Pre M(SD)':<16s} {'Post M(SD)':<16s} {'t':>8s} {'p':>10s} {'d':>8s} {'sig':>4s}"
    print(f"\n  {header}")
    print("  " + "-" * len(header))

    for band in bands_plus:
        pre_col = f"pre_{band}"
        post_col = f"post_{band}"

        if pre_col not in stress.columns or post_col not in stress.columns:
            continue

        pre = stress[pre_col].dropna()
        post = stress[post_col].dropna()

        # Paired: need both present
        mask = stress[pre_col].notna() & stress[post_col].notna()
        paired_pre = stress.loc[mask, pre_col]
        paired_post = stress.loc[mask, post_col]

        if len(paired_pre) < 3:
            continue

        # Check normality of differences
        diff = paired_pre.values - paired_post.values
        if len(diff) >= 8:
            _, norm_p = stats.shapiro(diff)
        else:
            norm_p = 1.0  # assume normal for tiny samples

        # Use parametric or non-parametric
        if norm_p > 0.05:
            t_stat, p_val = stats.ttest_rel(paired_pre, paired_post)
            test_name = "t"
        else:
            t_stat, p_val = stats.wilcoxon(paired_pre, paired_post)
            test_name = "W"

        d = cohens_d(paired_pre, paired_post)

        # Bonferroni correction
        p_adj = min(p_val * BONFERRONI_TESTS_RQ1, 1.0)

        print(f"  {band:<20s} {fmt(paired_pre.mean()):<8s}({fmt(paired_pre.std())})"
              f" {fmt(paired_post.mean()):<8s}({fmt(paired_post.std())})"
              f" {fmt(t_stat, 2):>8s} {fmt_p(p_adj):>10s} {fmt(d, 2):>8s} {stars(p_adj):>4s}")

        results[band] = {
            "pre_mean": float(paired_pre.mean()),
            "pre_sd": float(paired_pre.std()),
            "post_mean": float(paired_post.mean()),
            "post_sd": float(paired_post.std()),
            "test": test_name,
            "statistic": float(t_stat),
            "p_raw": float(p_val),
            "p_bonferroni": float(p_adj),
            "cohens_d": float(d),
            "n_pairs": int(len(paired_pre)),
        }

    # ── 3.3.2: Self-report correlations ───────────────────────────────────────

    print(f"\n  Self-Report Correlations (Spearman):")
    print(f"  {'Pair':<55s} {'rho':>8s} {'p':>10s}")
    print("  " + "-" * 75)

    corr_results = {}

    # Stress during vs work-phase stress (if we have a stress index)
    # The survey has 'stressed_during' and 'feeling_now'
    # Pre-post stress change: pre_alpha_beta_ratio - post_alpha_beta_ratio (higher ratio = more relaxed)
    stress["relaxation_change"] = stress["post_alpha_beta_ratio"] - stress["pre_alpha_beta_ratio"]

    corr_pairs = [
        ("stressed_during", "pre_beta", "Stress during vs. mean pre beta"),
        ("feeling_now", "post_alpha", "Feeling now vs. post alpha"),
    ]

    for survey_col, eeg_col, label in corr_pairs:
        if survey_col in stress.columns and eeg_col in stress.columns:
            rho, p = safe_corr(stress[survey_col], stress[eeg_col])
            print(f"  {label:<55s} {fmt(rho, 2):>8s} {fmt_p(p):>10s}")
            corr_results[label] = {"rho": rho, "p": p}

    # Breathing helpfulness vs relaxation change (only intervention-triggered)
    triggered = stress[stress["intervention_triggered"] == True]
    if "breathing_helped" in triggered.columns and len(triggered) >= 3:
        rho, p = safe_corr(triggered["breathing_helped"], triggered["relaxation_change"])
        label = "Breathing helped vs. relaxation change (triggered only)"
        print(f"  {label:<55s} {fmt(rho, 2):>8s} {fmt_p(p):>10s}")
        corr_results[label] = {"rho": rho, "p": p}

    # ── 3.3.3: Intervention subgroup ──────────────────────────────────────────

    print(f"\n  Intervention Subgroup Analysis:")
    not_triggered = stress[stress["intervention_triggered"] == False]

    for label, sub in [("Triggered", triggered), ("Not triggered", not_triggered)]:
        if len(sub) > 0:
            pre_s = sub["pre_beta"].mean() / (sub["pre_alpha"].mean() + 1e-9)
            post_s = sub["post_beta"].mean() / (sub["post_alpha"].mean() + 1e-9)
            print(f"  {label:<20s} n={len(sub):>3d}  "
                  f"Pre B/A ratio={fmt(pre_s, 3)}  Post B/A ratio={fmt(post_s, 3)}  "
                  f"Change={fmt(post_s - pre_s, 3)}")

    return {"n": n, "bands": results, "correlations": corr_results}


# ── Section 3.4: RQ2 — Food and EEG ──────────────────────────────────────────

def rq2_food(df: pd.DataFrame) -> dict:
    """RQ2: Pre-meal vs post-meal band powers for food sessions."""
    food = df[df["block_type"] == "food"].copy()
    n = len(food)

    print("\n" + "=" * 60)
    print(f"TABLE 4: RQ2 — Food Session EEG (n={n})")
    print("Pre-Meal vs. Post-Meal Comparison")
    print("=" * 60)

    if n < 3:
        print("  [SKIP] Too few food sessions for analysis")
        return {"n": n, "skip": True}

    results = {}
    bands_plus = EEG_BANDS + ["alpha_beta_ratio", "theta_beta_ratio"]

    header = f"{'Band':<20s} {'Pre M(SD)':<16s} {'Post M(SD)':<16s} {'t':>8s} {'p':>10s} {'d':>8s} {'sig':>4s}"
    print(f"\n  {header}")
    print("  " + "-" * len(header))

    for band in bands_plus:
        pre_col = f"pre_{band}"
        post_col = f"post_{band}"

        if pre_col not in food.columns or post_col not in food.columns:
            continue

        mask = food[pre_col].notna() & food[post_col].notna()
        paired_pre = food.loc[mask, pre_col]
        paired_post = food.loc[mask, post_col]

        if len(paired_pre) < 3:
            continue

        diff = paired_pre.values - paired_post.values
        if len(diff) >= 8:
            _, norm_p = stats.shapiro(diff)
        else:
            norm_p = 1.0

        if norm_p > 0.05:
            t_stat, p_val = stats.ttest_rel(paired_pre, paired_post)
            test_name = "t"
        else:
            t_stat, p_val = stats.wilcoxon(paired_pre, paired_post)
            test_name = "W"

        d = cohens_d(paired_pre, paired_post)
        p_adj = min(p_val * BONFERRONI_TESTS_RQ2, 1.0)

        print(f"  {band:<20s} {fmt(paired_pre.mean()):<8s}({fmt(paired_pre.std())})"
              f" {fmt(paired_post.mean()):<8s}({fmt(paired_post.std())})"
              f" {fmt(t_stat, 2):>8s} {fmt_p(p_adj):>10s} {fmt(d, 2):>8s} {stars(p_adj):>4s}")

        results[band] = {
            "pre_mean": float(paired_pre.mean()),
            "pre_sd": float(paired_pre.std()),
            "post_mean": float(paired_post.mean()),
            "post_sd": float(paired_post.std()),
            "test": test_name,
            "statistic": float(t_stat),
            "p_raw": float(p_val),
            "p_bonferroni": float(p_adj),
            "cohens_d": float(d),
            "n_pairs": int(len(paired_pre)),
        }

    # ── 3.4.2: Self-report and EEG correlations ──────────────────────────────

    print(f"\n  Self-Report vs. EEG Correlations (Spearman):")
    print(f"  {'Pair':<55s} {'rho':>8s} {'p':>10s}")
    print("  " + "-" * 75)

    corr_results = {}

    food["alpha_change"] = food["post_alpha"] - food["pre_alpha"]
    food["theta_change"] = food["post_theta"] - food["pre_theta"]
    food["beta_change"] = food["post_beta"] - food["pre_beta"]

    corr_pairs = [
        ("pre_hunger", "pre_theta", "Pre-hunger vs. pre-meal theta"),
        ("food_healthy", "alpha_change", "Food healthiness vs. alpha change"),
        ("post_energy", "post_beta", "Post-energy vs. post-meal beta"),
        ("post_mood", "post_alpha_beta_ratio", "Post-mood vs. post alpha/beta ratio"),
        ("post_satisfied", "alpha_change", "Post-satisfied vs. alpha change"),
    ]

    for survey_col, eeg_col, label in corr_pairs:
        if survey_col in food.columns and eeg_col in food.columns:
            rho, p = safe_corr(food[survey_col], food[eeg_col])
            print(f"  {label:<55s} {fmt(rho, 2):>8s} {fmt_p(p):>10s}")
            corr_results[label] = {"rho": float(rho) if not np.isnan(rho) else None,
                                   "p": float(p) if not np.isnan(p) else None}

    # ── 3.4.3: Regression: post-meal mood ─────────────────────────────────────

    print(f"\n  Regression: Post-Meal Mood ~ Predictors")

    reg_cols = ["pre_hunger", "food_healthy", "alpha_change", "theta_change"]
    target = "post_mood"

    if target in food.columns and all(c in food.columns for c in reg_cols):
        reg_data = food[[target] + reg_cols].dropna()
        if len(reg_data) >= 10:
            from sklearn.linear_model import LinearRegression
            X = reg_data[reg_cols].values
            y = reg_data[target].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"    R-squared: {r2:.3f}  (n={len(reg_data)})")
            print(f"    {'Predictor':<20s} {'B':>8s}")
            print(f"    " + "-" * 30)
            for name, coef in zip(reg_cols, model.coef_):
                print(f"    {name:<20s} {coef:>8.4f}")
            print(f"    {'Intercept':<20s} {model.intercept_:>8.4f}")
        else:
            print(f"    [SKIP] Only {len(reg_data)} complete rows — need >= 10")
    else:
        print(f"    [SKIP] Missing columns for regression")

    return {"n": n, "bands": results, "correlations": corr_results}


# ── Section 3.5: Cross-session comparison ─────────────────────────────────────

def cross_session(df: pd.DataFrame) -> dict:
    """Cross-session within-subject consistency."""
    print("\n" + "=" * 60)
    print("TABLE 5: Cross-Session Consistency")
    print("=" * 60)

    # Participants with both sessions
    both = df.groupby("participant_code").filter(lambda g: g["block_type"].nunique() == 2)
    codes = both["participant_code"].unique()
    n = len(codes)

    if n < 3:
        print(f"  [SKIP] Only {n} participants with both sessions")
        return {"n": n, "skip": True}

    print(f"  Participants with both sessions: {n}")

    results = {}
    for band in ["alpha", "beta", "theta"]:
        pre_col = f"pre_{band}"
        stress_vals = both[both["block_type"] == "stress"].set_index("participant_code")[pre_col]
        food_vals = both[both["block_type"] == "food"].set_index("participant_code")[pre_col]
        common = stress_vals.index.intersection(food_vals.index)
        if len(common) >= 3:
            rho, p = stats.spearmanr(stress_vals[common], food_vals[common])
            print(f"  Baseline {band:<8s} across sessions: rho={fmt(rho, 2)}, p={fmt_p(p)} (n={len(common)})")
            results[f"baseline_{band}"] = {"rho": float(rho), "p": float(p), "n": len(common)}

    return {"n": n, "results": results}


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, outdir: Path):
    """Generate publication-ready figures."""
    if not HAS_PLOTS:
        return

    outdir.mkdir(exist_ok=True)

    # ── Figure 1: Pre vs Post band powers (stress) ────────────────────────────

    stress = df[df["block_type"] == "stress"]
    if len(stress) >= 3:
        fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=False)
        fig.suptitle("Figure 1: Stress Session — Baseline vs. Post-Session EEG Band Powers", fontsize=12)

        for i, band in enumerate(EEG_BANDS):
            ax = axes[i]
            pre = stress[f"pre_{band}"].dropna()
            post = stress[f"post_{band}"].dropna()
            positions = [1, 2]
            bp = ax.boxplot([pre, post], positions=positions, widths=0.6,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
            bp["boxes"][0].set_facecolor("#4C72B0")
            bp["boxes"][1].set_facecolor("#DD8452")
            ax.set_xticks(positions)
            ax.set_xticklabels(["Baseline", "Post"])
            ax.set_title(band.capitalize(), fontsize=10)
            if i == 0:
                ax.set_ylabel("Power (normalized)")

        plt.tight_layout()
        fig.savefig(outdir / "fig1_stress_pre_post.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  [PLOT] Saved: {outdir / 'fig1_stress_pre_post.png'}")

    # ── Figure 2: Pre vs Post band powers (food) ─────────────────────────────

    food = df[df["block_type"] == "food"]
    if len(food) >= 3:
        fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=False)
        fig.suptitle("Figure 2: Food Session — Pre-Meal vs. Post-Meal EEG Band Powers", fontsize=12)

        for i, band in enumerate(EEG_BANDS):
            ax = axes[i]
            pre = food[f"pre_{band}"].dropna()
            post = food[f"post_{band}"].dropna()
            bp = ax.boxplot([pre, post], positions=[1, 2], widths=0.6,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
            bp["boxes"][0].set_facecolor("#4C72B0")
            bp["boxes"][1].set_facecolor("#55A868")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Pre-Meal", "Post-Meal"])
            ax.set_title(band.capitalize(), fontsize=10)
            if i == 0:
                ax.set_ylabel("Power (normalized)")

        plt.tight_layout()
        fig.savefig(outdir / "fig2_food_pre_post.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PLOT] Saved: {outdir / 'fig2_food_pre_post.png'}")

    # ── Figure 3: Stress index — intervention triggered vs not ────────────────

    if len(stress) >= 3 and "intervention_triggered" in stress.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle("Figure 3: Alpha/Beta Ratio by Intervention Status", fontsize=12)

        groups = {
            "Triggered": stress[stress["intervention_triggered"] == True],
            "Not Triggered": stress[stress["intervention_triggered"] == False],
        }

        x_pos = 0
        colors = {"Triggered": "#C44E52", "Not Triggered": "#4C72B0"}
        for label, sub in groups.items():
            if len(sub) == 0:
                continue
            pre_ratio = sub["pre_alpha_beta_ratio"].dropna()
            post_ratio = sub["post_alpha_beta_ratio"].dropna()
            ax.bar(x_pos, pre_ratio.mean(), width=0.35, color=colors[label], alpha=0.6, label=f"{label} — Baseline")
            ax.bar(x_pos + 0.4, post_ratio.mean(), width=0.35, color=colors[label], alpha=1.0, label=f"{label} — Post")
            ax.errorbar(x_pos, pre_ratio.mean(), yerr=pre_ratio.std(), fmt="none", color="black", capsize=3)
            ax.errorbar(x_pos + 0.4, post_ratio.mean(), yerr=post_ratio.std(), fmt="none", color="black", capsize=3)
            x_pos += 1.2

        ax.set_ylabel("Alpha/Beta Ratio")
        ax.set_xticks([0.2, 1.4])
        ax.set_xticklabels(["Triggered", "Not Triggered"])
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(outdir / "fig3_intervention_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PLOT] Saved: {outdir / 'fig3_intervention_comparison.png'}")

    # ── Figure 4: Correlation scatter — survey vs EEG ─────────────────────────

    scatter_pairs = []
    if len(stress) >= 5:
        if "stressed_during" in stress.columns:
            scatter_pairs.append(("stressed_during", "pre_beta", stress, "Stress Session: Self-Reported Stress vs. Pre Beta"))
    if len(food) >= 5:
        if "pre_hunger" in food.columns:
            scatter_pairs.append(("pre_hunger", "pre_theta", food, "Food Session: Pre-Hunger vs. Pre-Meal Theta"))

    if scatter_pairs:
        fig, axes = plt.subplots(1, len(scatter_pairs), figsize=(6 * len(scatter_pairs), 4))
        if len(scatter_pairs) == 1:
            axes = [axes]

        for i, (x_col, y_col, data, title) in enumerate(scatter_pairs):
            ax = axes[i]
            mask = data[x_col].notna() & data[y_col].notna()
            x = data.loc[mask, x_col]
            y = data.loc[mask, y_col]
            ax.scatter(x, y, alpha=0.6, edgecolors="white", linewidth=0.5)
            if len(x) >= 3:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=1.5)
                rho, pv = stats.spearmanr(x, y)
                ax.set_title(f"{title}\nrho={rho:.2f}, p={fmt_p(pv)}", fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
            ax.set_xlabel(x_col.replace("_", " ").title())
            ax.set_ylabel(y_col.replace("_", " ").title())

        plt.tight_layout()
        fig.savefig(outdir / "fig4_correlations.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PLOT] Saved: {outdir / 'fig4_correlations.png'}")

    # ── Figure 5: Data quality distribution ───────────────────────────────────

    if "data_quality_score" in df.columns and df["data_quality_score"].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for block, color in [("stress", "#C44E52"), ("food", "#55A868")]:
            sub = df[df["block_type"] == block]["data_quality_score"].dropna()
            if len(sub) > 0:
                ax.hist(sub, bins=10, alpha=0.6, color=color, label=f"{block.capitalize()} (n={len(sub)})", edgecolor="white")
        ax.set_xlabel("Data Quality Score (0-100)")
        ax.set_ylabel("Count")
        ax.set_title("Figure 5: Data Quality Score Distribution")
        ax.axvline(MIN_QUALITY, color="red", linestyle="--", alpha=0.5, label=f"Min threshold ({MIN_QUALITY})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / "fig5_quality_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PLOT] Saved: {outdir / 'fig5_quality_distribution.png'}")


# ── Summary JSON ──────────────────────────────────────────────────────────────

def save_results(results: dict, outpath: Path):
    """Save all computed statistics to JSON for reference."""
    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(outpath, "w") as f:
        json.dump(convert(results), f, indent=2, default=str)
    print(f"\n  [JSON] Saved: {outpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_study_data.py <path-to-csv>")
        print("  Download CSV from: https://dream-analysis.vercel.app/study/admin")
        print("  (Click 'Export CSV' button)")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    print(f"Neural Dream Workshop — Pilot Study Analysis")
    print(f"Input: {csv_path}")
    print(f"{'=' * 60}")

    # Load and filter
    df = load_data(csv_path)
    df_clean = filter_quality(df)

    # Run all analyses
    all_results = {}
    all_results["demographics"] = demographics(df_clean)
    all_results["data_quality"] = data_quality(df_clean)
    all_results["rq1_stress"] = rq1_stress(df_clean)
    all_results["rq2_food"] = rq2_food(df_clean)
    all_results["cross_session"] = cross_session(df_clean)

    # Plots
    fig_dir = Path(csv_path).parent / "figures"
    make_plots(df_clean, fig_dir)

    # Save JSON
    json_path = Path(csv_path).with_suffix(".results.json")
    save_results(all_results, json_path)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete.")
    print(f"  - Copy table output above into paper Section 3")
    print(f"  - Figures saved to: {fig_dir}/")
    print(f"  - Full results JSON: {json_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
