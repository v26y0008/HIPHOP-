"""
win_loss_analysis.py (v2 追加モジュール)

勝敗（外部メディア評価ベース）による言説パターンの差異を分析する。

分析軸:
  1. 勝者側のビーフ曲 vs 敗者側のビーフ曲 でコメント語彙・熱量が異なるか
  2. 決着後（POST曲）に勝者側・敗者側でコメントパターンが異なるか
  3. B6（disputed）を含む場合と除く場合で結論が変わるかの感度分析
  4. B5（legalな決着=ラップバトルではない）を含む場合と除く場合の感度分析

勝敗ラベルは外部メディア評価（Rolling Stone, Pitchfork, YouGov調査等）に基づく。
根拠は outputs/related_work_memo.md に記載。

使い方:
  python win_loss_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from scipy.stats.contingency import association

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conflict_vocab_analysis import has_conflict_vocab, has_opponent_mention  # noqa: E402

BEEF_OUTCOME = {
    "b1": {"winner": "kendrick", "loser": "drake", "verdict": "clear", "outcome_type": "rap_battle"},
    "b2": {"winner": "pusha", "loser": "drake", "verdict": "clear", "outcome_type": "rap_battle"},
    "b4": {"winner": "eminem", "loser": "mgk", "verdict": "clear", "outcome_type": "rap_battle"},
    "b5": {"winner": "wayne", "loser": "birdman", "verdict": "clear", "outcome_type": "legal"},
    "b6": {"winner": "megan", "loser": "nicki", "verdict": "disputed", "outcome_type": "rap_battle"},
}
# B7(Ice Cube vs NWA)追加を試みたが、両アーティストのbeefカテゴリ動画
# (No Vaseline, Real Niggaz)ともコメント欄無効(403)のためbuild_dataset.pyの
# EXCLUDED_BEEFSで除外済み。参考: 勝敗の実質的な評価はIce Cube側の圧勝が
# 批評的コンセンサス(NWA側は正式な返答曲を出せずグループ解散に至った)。


def assign_win_loss(df):
    def get_win_loss(row):
        outcome = BEEF_OUTCOME.get(row["beef_id"], {})
        side = str(row["artist_side"]).lower()
        if side == outcome.get("winner", ""):
            return "winner"
        if side == outcome.get("loser", ""):
            return "loser"
        return "unknown"

    df = df.copy()
    df["win_loss_side"] = df.apply(get_win_loss, axis=1)
    df["verdict"] = df["beef_id"].map(lambda b: BEEF_OUTCOME.get(b, {}).get("verdict", "unknown"))
    df["outcome_type"] = df["beef_id"].map(lambda b: BEEF_OUTCOME.get(b, {}).get("outcome_type", "unknown"))
    return df


def run_win_loss_analysis(df, label):
    print(f"\n{'=' * 60}\n=== 勝敗分析: {label} ===\n{'=' * 60}")

    beef_only = df[df["track_category"] == "beef"]
    winner_df = beef_only[beef_only["win_loss_side"] == "winner"]
    loser_df = beef_only[beef_only["win_loss_side"] == "loser"]

    print(f"勝者側コメント数: {len(winner_df):,}  /  敗者側コメント数: {len(loser_df):,}")

    if len(winner_df) == 0 or len(loser_df) == 0:
        print("  勝者側/敗者側どちらかのデータが0件のためスキップ")
        return pd.DataFrame()

    results = []
    combined = pd.concat([winner_df, loser_df])

    for target_col, target_name in [("has_conflict", "対立語彙率"), ("has_opponent", "相手名言及率")]:
        ct = pd.crosstab(combined["win_loss_side"], combined[target_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            print(f"\n[{target_name}] 分割表が2x2未満のためスキップ")
            continue
        chi2, p, dof, expected = chi2_contingency(ct)
        v = association(ct, method="cramer")
        n_low = int((expected < 5).sum())

        winner_pct = winner_df[target_col].mean() * 100
        loser_pct = loser_df[target_col].mean() * 100

        print(f"\n[{target_name}]")
        print(f"  勝者側: {winner_pct:.2f}%  敗者側: {loser_pct:.2f}%  差分: {winner_pct - loser_pct:+.2f}pp")
        print(f"  chi2={chi2:.3f}, p={p:.4e}, V={v:.3f}" + (f"  !!期待度数<5:{n_low}件!!" if n_low else ""))

        results.append({
            "label": label, "target": target_name,
            "winner_pct": winner_pct, "loser_pct": loser_pct, "diff_pp": winner_pct - loser_pct,
            "chi2": chi2, "p_value": p, "cramers_v": v, "significant": p < 0.05,
            "n_low_expected_cells": n_low,
        })

    print("\n[熱量指標: Mann-Whitney U検定]")
    for metric, metric_name in [("like_count", "いいね数"), ("reply_count", "返信数")]:
        w_vals = winner_df[metric].dropna()
        l_vals = loser_df[metric].dropna()
        if len(w_vals) == 0 or len(l_vals) == 0:
            continue
        stat, p = mannwhitneyu(w_vals, l_vals, alternative="two-sided")
        print(f"  {metric_name}: 勝者側中央値={w_vals.median():.1f}, 敗者側中央値={l_vals.median():.1f}, p={p:.4e}")
        results.append({
            "label": label, "target": metric_name,
            "winner_pct": w_vals.median(), "loser_pct": l_vals.median(), "diff_pp": None,
            "chi2": None, "p_value": p, "cramers_v": None, "significant": p < 0.05,
            "n_low_expected_cells": None,
        })

    return pd.DataFrame(results)


def run_post_win_loss_analysis(df):
    print(f"\n{'=' * 60}\n=== POST曲における勝敗別コメントパターン ===\n{'=' * 60}")
    post_df = df[df["track_category"] == "post"]
    if len(post_df) == 0:
        print("  POST曲のデータがありません")
        return

    for bid in sorted(post_df["beef_id"].unique()):
        outcome = BEEF_OUTCOME.get(bid, {})
        winner, loser = outcome.get("winner", ""), outcome.get("loser", "")
        b_post = post_df[post_df["beef_id"] == bid]

        print(f"\n{bid} ({winner}[勝] vs {loser}[敗]):")
        for side, label in [(winner, "勝者"), (loser, "敗者")]:
            sub = b_post[b_post["artist_side"] == side]
            if len(sub) == 0:
                print(f"  {label}側（{side}）: データなし")
                continue
            conflict_pct = sub["has_conflict"].mean() * 100
            opponent_pct = sub["has_opponent"].mean() * 100
            print(f"  {label}側（{side}）: n={len(sub):,}, 対立語彙={conflict_pct:.1f}%, 相手名={opponent_pct:.1f}%")


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に preprocess.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df["has_conflict"] = df["text_clean"].fillna("").apply(has_conflict_vocab)
    df["has_opponent"] = df.apply(
        lambda r: has_opponent_mention(r["text_clean"], r["beef_id"], r["artist_side"]), axis=1
    )
    df = assign_win_loss(df)

    results_all = run_win_loss_analysis(df, "全5ビーフ（B6 disputed含む）")
    results_clear = run_win_loss_analysis(df[df["verdict"] == "clear"], "clear verdictのみ（B6除外）")
    results_rb = run_win_loss_analysis(df[df["outcome_type"] == "rap_battle"], "rap_battle型のみ（B5除外）")
    run_post_win_loss_analysis(df)

    all_results = pd.concat([results_all, results_clear, results_rb], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(OUT_DIR / "win_loss_results_v2.csv", index=False)
    df.to_csv(OUT_DIR / "comments_with_win_loss_v2.csv", index=False)
    print(f"\n結果を保存しました -> {OUT_DIR / 'win_loss_results_v2.csv'}, {OUT_DIR / 'comments_with_win_loss_v2.csv'}")


if __name__ == "__main__":
    main()
