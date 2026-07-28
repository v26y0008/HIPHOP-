"""
conflict_vocab_analysis.py (v2 設計)

比較軸：カタログ曲 vs ビーフ曲 vs ポスト曲（track_category）
data/processed/comments_clean_v2.csv を使い、beef_id×track_category別の
対立語彙率・相手アーティスト名言及率を集計し、カイ二乗検定・Cramer's V・
Bonferroni補正を行う。

B3(Drake vs Meek Mill)・B7(Ice Cube vs NWA)はbeefカテゴリのコメントが
両アーティストともYouTube側でコメント欄無効のため、build_dataset.py の
時点で既に除外済み（したがってこのファイルのOPPONENT_NAMES/BEEF_LABELSに
b3・b7は登場しない）。

使い方:
  python conflict_vocab_analysis.py
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
STATS_DIR = BEEF_ROOT / "outputs" / "stats"

CATEGORY_ORDER = ["catalog", "beef", "post"]
ALPHA = 0.05

CONFLICT_VOCAB = [
    "diss", "beef", "dissed", "destroyed", "bodied", "ate",
    "cooked", "buried", "exposed", "ended", "won", "lost",
    "winner", "loser", "battle", "respond", "response",
    "shot", "fired", "bars", "clapped", "murked",
]

# ビーフごとの相手アーティスト名（artist_sideで管理）
OPPONENT_NAMES = {
    "b1": {"kendrick": ["drake", "aubrey", "drizzy", "champagne papi"],
           "drake": ["kendrick", "kdot", "k dot", "pglang", "dot"]},
    "b2": {"pusha": ["drake", "aubrey", "drizzy"],
           "drake": ["pusha", "pusha t", "king push", "clipse"]},
    "b4": {"eminem": ["mgk", "machine gun kelly", "kells"],
           "mgk": ["eminem", "em", "slim shady", "marshall"]},
    "b5": {"wayne": ["birdman", "baby", "cash money", "cm"],
           "birdman": ["wayne", "weezy", "tunechi"]},
    "b6": {"megan": ["nicki", "nicki minaj", "barbie"],
           "nicki": ["megan", "meg", "thee stallion"]},
}

BEEF_LABELS = {
    "b1": "B1: Kendrick vs Drake",
    "b2": "B2: Pusha T vs Drake",
    "b4": "B4: Eminem vs MGK",
    "b5": "B5: Wayne vs Birdman",
    "b6": "B6: Megan vs Nicki",
}


def compile_pattern(words):
    escaped = [re.escape(w) for w in words]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


CONFLICT_PATTERN = compile_pattern(CONFLICT_VOCAB)


def has_conflict_vocab(text):
    return bool(CONFLICT_PATTERN.search(str(text)))


def has_opponent_mention(text, beef_id, artist_side):
    names = OPPONENT_NAMES.get(beef_id, {}).get(artist_side)
    if not names:
        return False
    pattern = compile_pattern(names)
    return bool(pattern.search(str(text)))


def cramers_v(chi2, n, r, c):
    return math.sqrt(chi2 / (n * (min(r, c) - 1)))


def run_chi2(df_beef, target_col):
    ct = df_beef.groupby("track_category")[target_col].agg(["sum", "count"])
    ct = ct.reindex(CATEGORY_ORDER).dropna()
    if len(ct) < 2:
        return None

    pos = ct["sum"].to_numpy()
    neg = (ct["count"] - ct["sum"]).to_numpy()
    table = np.array([pos, neg])
    chi2, p, dof, expected = chi2_contingency(table)
    n_total = int(ct["count"].sum())
    v = cramers_v(chi2, n_total, table.shape[0], table.shape[1])
    n_low_expected = int((expected < 5).sum())

    return {
        "chi2": chi2, "p_value": p, "dof": dof, "cramers_v": v,
        "n_low_expected_cells": n_low_expected,
        "categories": list(ct.index),
        "sample_sizes": ct["count"].astype(int).to_dict(),
    }


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に preprocess.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df["has_conflict"] = df["text_clean"].fillna("").apply(has_conflict_vocab)
    df["has_opponent"] = df.apply(
        lambda r: has_opponent_mention(r["text_clean"], r["beef_id"], r["artist_side"]), axis=1
    )

    # ============================================================
    # 分析1: beef_id x track_category 別の語彙出現率サマリー
    # ============================================================
    summary = df.groupby(["beef_id", "track_category"]).agg(
        n_comments=("comment_id", "count"),
        n_conflict=("has_conflict", "sum"),
        pct_conflict=("has_conflict", "mean"),
        n_opponent=("has_opponent", "sum"),
        pct_opponent=("has_opponent", "mean"),
        avg_length=("text_clean", lambda x: x.str.len().mean()),
        reply_rate=("reply_count", "mean"),
        like_rate=("like_count", "mean"),
    ).reset_index()
    summary["pct_conflict"] *= 100
    summary["pct_opponent"] *= 100

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary_by_beef_category.csv"
    summary.to_csv(summary_path, index=False)
    print(f"beef_id x track_category サマリー -> {summary_path}")
    print(summary.to_string(index=False))

    # ============================================================
    # 分析2: ビーフごとのカイ二乗検定（対立語彙率・相手名言及率）
    # ============================================================
    results = []
    for bid in sorted(df["beef_id"].unique()):
        df_b = df[df["beef_id"] == bid]
        for target, col in [("conflict_vocab", "has_conflict"), ("opponent_mention", "has_opponent")]:
            res = run_chi2(df_b, col)
            if res is None:
                continue
            res.update({
                "beef_id": bid, "beef_name": BEEF_LABELS.get(bid, bid),
                "beef_type": df_b["beef_type"].iloc[0],
                "drake_involved": df_b["drake_involved"].iloc[0],
                "target": target,
            })
            results.append(res)

    n_tests = len(results)
    alpha_corrected = ALPHA / n_tests if n_tests else ALPHA
    for r in results:
        r["significant_raw"] = r["p_value"] < ALPHA
        r["significant_bonferroni"] = r["p_value"] < alpha_corrected
        r["alpha_corrected"] = alpha_corrected

    print(f"\n=== カイ二乗検定結果（Bonferroni補正: 検定数={n_tests}, α={alpha_corrected:.4f}） ===")
    for r in results:
        print(
            f"{r['beef_name']} / {r['target']}: chi2={r['chi2']:.2f}, p={r['p_value']:.4e}, "
            f"V={r['cramers_v']:.3f}, 補正後有意={r['significant_bonferroni']}"
            + (f"  !!期待度数<5のセル{r['n_low_expected_cells']}件!!" if r["n_low_expected_cells"] else "")
        )

    results_df = pd.DataFrame(results)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    chi2_path = STATS_DIR / "chi2_results_v2.csv"
    results_df.to_csv(chi2_path, index=False)
    print(f"\nカイ二乗検定結果 -> {chi2_path}")

    # ============================================================
    # 分析3: Drake関与 vs 非関与
    # ============================================================
    print("\n=== Drake関与 vs 非関与 ===")
    for drake in [True, False]:
        subset = results_df[results_df["drake_involved"] == drake]
        n_sig = subset["significant_bonferroni"].sum()
        label = "Drake関与" if drake else "非Drake"
        print(f"{label}: {n_sig}/{len(subset)} 件が補正後有意")

    # ============================================================
    # 分析4: ビーフタイプ別
    # ============================================================
    print("\n=== ビーフタイプ別 ===")
    for btype in results_df["beef_type"].unique():
        subset = results_df[results_df["beef_type"] == btype]
        n_sig = subset["significant_bonferroni"].sum()
        print(f"{btype}: {n_sig}/{len(subset)} 件が補正後有意")


if __name__ == "__main__":
    main()
