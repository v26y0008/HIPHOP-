"""
merge_llm_stance_result.py (v2 追加モジュール、Task5仕上げ)

build_llm_stance_sample.py で生成したプロンプトを別のClaude会話に投げて得られた
`sample_id,stance` のCSVを data/processed/llm_stance_result.csv として保存した後、
本体データ(llm_stance_sample.csv)と結合し、対立語彙分析・勝敗分析と整合するかを検証する。

事前準備:
  1. outputs/llm_stance_prompt.md の中身を新しいClaude会話に貼り付ける
  2. 返ってきたCSV（sample_id,stance の2列）を
     data/processed/llm_stance_result.csv として保存する
  3. このスクリプトを実行する

使い方:
  python merge_llm_stance_result.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

BEEF_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = BEEF_ROOT / "data" / "processed" / "llm_stance_sample.csv"
RESULT_CSV = BEEF_ROOT / "data" / "processed" / "llm_stance_result.csv"
OUT_CSV = BEEF_ROOT / "data" / "processed" / "llm_stance_merged.csv"

# 各ビーフの勝者側(artist_side)。win_loss_analysis.py の BEEF_OUTCOME と一致させる
WINNER_SIDE = {"b1": "a", "b2": "a", "b4": "a", "b5": "a", "b6": "a"}
# a=artist_a が勝者のケースのみ（本プロジェクトの5ビーフは全てartist_a=勝者になるよう
# build_llm_stance_sample.py のBEEF_INFOで順序を設定済み: Kendrick/PushaT/Eminem/Wayne/Megan）


def main():
    if not RESULT_CSV.exists():
        print(f"エラー: {RESULT_CSV} が見つかりません。")
        print("先にoutputs/llm_stance_prompt.mdを別のClaude会話に貼り付け、")
        print(f"返ってきたCSV(sample_id,stance)を {RESULT_CSV} として保存してください。")
        return

    sample = pd.read_csv(SAMPLE_CSV, keep_default_na=False, na_values=[""])
    result = pd.read_csv(RESULT_CSV, keep_default_na=False, na_values=[""])

    merged = sample.merge(result, on="sample_id", how="left")
    n_missing = merged["stance"].isna().sum()
    print(f"結合件数: {len(merged)} (stance欠損: {n_missing})")

    merged["stance"] = merged["stance"].str.strip().str.lower()
    merged["winner_side"] = merged["beef_id"].map(WINNER_SIDE)
    merged["stance_toward"] = merged.apply(
        lambda r: "winner" if r["stance"] == r["winner_side"]
        else ("loser" if r["stance"] in ("a", "b") else "neutral"),
        axis=1,
    )

    print("\n=== beef_id別 stance分布 ===")
    print(merged.groupby(["beef_id", "stance"]).size().unstack(fill_value=0))

    print("\n=== 全体 stance_toward分布（勝者支持 vs 敗者支持 vs 中立）===")
    print(merged["stance_toward"].value_counts())

    # 対立語彙(has_conflict)を持つコメントは中立でなく明確な支持を示しやすいか
    wl_path = BEEF_ROOT / "data" / "processed" / "comments_with_win_loss_v2.csv"
    if wl_path.exists():
        wl = pd.read_csv(wl_path, keep_default_na=False, na_values=[""])
        merged2 = merged.merge(wl[["comment_id", "has_conflict"]], on="comment_id", how="left")
        ct = pd.crosstab(merged2["has_conflict"], merged2["stance_toward"])
        print("\n=== has_conflict x stance_toward クロス集計 ===")
        print(ct)
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, _ = chi2_contingency(ct)
            print(f"カイ二乗検定: chi2={chi2:.2f}, p={p:.4g}")

    merged.to_csv(OUT_CSV, index=False)
    print(f"\n保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
