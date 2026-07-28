"""
statistical_tests.py

data/processed/cross_beef_period_summary.csv を使い、main beef(beef1/2/3)の
PRE/PEAK/POST 3期間について、対立語彙率・相手アーティスト名言及率の
カイ二乗検定とCramer's Vを計算する。

3beef × 2指標 = 6回の検定を同一データに対して行うため、Bonferroni補正
（有意水準 0.05 / 検定数）による多重比較調整後の有意性も併せて報告する。

出力:
  outputs/stats/test_results.txt   人間が読む形式の結果
  outputs/stats/chi2_results.csv   generate_tables.py が読み込む機械可読形式

使い方:
  python statistical_tests.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BEEF_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = BEEF_ROOT / "data" / "processed" / "cross_beef_period_summary.csv"
OUTPUT_DIR = BEEF_ROOT / "outputs" / "stats"

MAIN_BEEFS = ["beef1", "beef2", "beef3"]
PERIOD_ORDER = ["PRE", "PEAK", "POST"]
SMALL_N_THRESHOLD = 50
ALPHA = 0.05

BEEF_LABELS = {
    "beef1": "Beef1: Kendrick vs Drake",
    "beef2": "Beef2: Pusha T vs Drake",
    "beef3": "Beef3: Wayne vs Birdman",
}


def cramers_v(chi2, n, r, c):
    return math.sqrt(chi2 / (n * (min(r, c) - 1)))


def run_test(df, beef_id, count_col, label):
    sub = df[df["beef_id"] == beef_id].set_index("period").reindex(PERIOD_ORDER)
    if sub["n_comments"].isna().any():
        return None

    n_comments = sub["n_comments"].to_numpy()
    n_positive = sub[count_col].to_numpy()
    n_negative = n_comments - n_positive

    table = np.array([n_positive, n_negative])
    chi2, p, dof, _ = chi2_contingency(table)
    n_total = int(n_comments.sum())
    v = cramers_v(chi2, n_total, table.shape[0], table.shape[1])

    return {
        "beef_id": beef_id,
        "test": label,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": v,
        "n_total": n_total,
        "n_pre": int(n_comments[0]),
        "n_peak": int(n_comments[1]),
        "n_post": int(n_comments[2]),
        "small_n_warning": bool(n_comments[0] < SMALL_N_THRESHOLD),
    }


def main():
    df = pd.read_csv(SUMMARY_CSV, keep_default_na=False, na_values=[""])

    results = []
    for beef_id in MAIN_BEEFS:
        r1 = run_test(df, beef_id, "n_conflict_vocab", "conflict_vocab")
        r2 = run_test(df, beef_id, "n_opponent_mention", "opponent_mention")
        for r in (r1, r2):
            if r is not None:
                results.append(r)

    # 多重比較補正: 3beef × 2指標 = 6回の検定に対してBonferroni補正をかける
    n_tests = len(results)
    alpha_corrected = ALPHA / n_tests if n_tests else ALPHA
    for r in results:
        r["significant_raw"] = r["p_value"] < ALPHA
        r["significant_bonferroni"] = r["p_value"] < alpha_corrected

    lines = []
    results_by_beef = {}
    for r in results:
        results_by_beef.setdefault(r["beef_id"], []).append(r)

    for beef_id in MAIN_BEEFS:
        lines.append(f"=== {BEEF_LABELS[beef_id]} ===")
        for r in results_by_beef.get(beef_id, []):
            jp_label = "対立語彙率" if r["test"] == "conflict_vocab" else "相手名言及率"
            lines.append(
                f"[{jp_label}] chi2={r['chi2']:.2f}, df={r['dof']}, "
                f"p={r['p_value']:.2e}, Cramer's V={r['cramers_v']:.2f} "
                f"(n_PRE={r['n_pre']}, n_PEAK={r['n_peak']}, n_POST={r['n_post']})"
            )
            bonf_status = "有意" if r["significant_bonferroni"] else "非有意"
            lines.append(
                f"  Bonferroni補正後(α={alpha_corrected:.4f}, 検定数={n_tests}): {bonf_status}"
            )
            if r["small_n_warning"]:
                lines.append(
                    f"  !! 警告: PRE期間のサンプルサイズが小さい(n={r['n_pre']}<{SMALL_N_THRESHOLD})ため解釈に注意"
                )
        lines.append("")

    lines.append(
        f"=== 多重比較補正まとめ（Bonferroni, 検定数={n_tests}, 補正後α={alpha_corrected:.4f}） ==="
    )
    for r in results:
        flip = r["significant_raw"] != r["significant_bonferroni"]
        note = "  !! 補正で結論が変わる !!" if flip else ""
        lines.append(
            f"{BEEF_LABELS[r['beef_id']]} / {r['test']}: p={r['p_value']:.2e} -> "
            f"補正前={'有意' if r['significant_raw'] else '非有意'}, "
            f"補正後={'有意' if r['significant_bonferroni'] else '非有意'}{note}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text_path = OUTPUT_DIR / "test_results.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"検定結果(テキスト) -> {text_path}")

    csv_path = OUTPUT_DIR / "chi2_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"検定結果(CSV) -> {csv_path}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
