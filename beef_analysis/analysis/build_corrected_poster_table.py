"""
build_corrected_poster_table.py

PPTX(hiphop_beef_poster_final.pptx)に混入した旧v1データ・捏造データを
実際のv2データで置き換えるための修正版表を生成する。

PPTXはビーフを連番(B1〜B5)で振り直しているため、本スクリプトの出力にも
PPTX表記の連番(poster_label)と実際のbeef_id(内部key)の両方を記載する。

出力:
  outputs/tables/corrected_poster_table.csv
  outputs/tables/corrected_poster_table.tex

使い方:
  python build_corrected_poster_table.py
"""

from pathlib import Path

import pandas as pd

BEEF_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = BEEF_ROOT / "data" / "processed" / "summary_by_beef_category.csv"
CHI2_PATH = BEEF_ROOT / "outputs" / "stats" / "chi2_results_v2.csv"
TABLES_DIR = BEEF_ROOT / "outputs" / "tables"

# PPTXでの連番(B1〜B5) <-> 本パイプラインの内部beef_id の対応
POSTER_LABEL = {
    "b1": "B1 (Kendrick Lamar vs Drake)",
    "b2": "B2 (Pusha T vs Drake)",
    "b4": "B3 (Eminem vs MGK) ※PPTXのB3は本パイプラインのb4",
    "b5": "B4 (Wayne vs Birdman) ※PPTXのB4は本パイプラインのb5",
    "b6": "B5 (Megan vs Nicki) ※PPTXのB5は本パイプラインのb6",
}
BEEF_ORDER = ["b1", "b2", "b4", "b5", "b6"]
CATEGORY_ORDER = ["catalog", "beef", "post"]


def main():
    summary = pd.read_csv(SUMMARY_PATH)
    chi2 = pd.read_csv(CHI2_PATH)

    # ---- 表1: カタログ/ビーフ/POST 別の対立語彙率・相手名言及率 ----
    rows = []
    for bid in BEEF_ORDER:
        row = {"poster_label": POSTER_LABEL[bid], "internal_beef_id": bid}
        for cat in CATEGORY_ORDER:
            sub = summary[(summary["beef_id"] == bid) & (summary["track_category"] == cat)]
            if len(sub):
                row[f"conflict_pct_{cat}"] = round(sub["pct_conflict"].iloc[0], 2)
                row[f"opponent_pct_{cat}"] = round(sub["pct_opponent"].iloc[0], 2)
                row[f"n_comments_{cat}"] = int(sub["n_comments"].iloc[0])
        rows.append(row)
    table1 = pd.DataFrame(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table1_path = TABLES_DIR / "corrected_poster_table.csv"
    table1.to_csv(table1_path, index=False)
    print(f"表1(カテゴリ別出現率) -> {table1_path}")
    print(table1.to_string(index=False))

    # ---- 表2: カイ二乗検定結果(対立語彙率・相手名言及率、Bonferroni補正込み) ----
    chi2_out = chi2.copy()
    chi2_out["poster_label"] = chi2_out["beef_id"].map(POSTER_LABEL)
    chi2_out = chi2_out[[
        "poster_label", "beef_id", "target", "chi2", "p_value", "cramers_v",
        "significant_bonferroni",
    ]].rename(columns={
        "beef_id": "internal_beef_id", "target": "test_target",
        "cramers_v": "cramers_v", "significant_bonferroni": "significant_after_bonferroni",
    })
    chi2_out = chi2_out.sort_values(["internal_beef_id", "test_target"])

    table2_path = TABLES_DIR / "corrected_poster_chi2.csv"
    chi2_out.to_csv(table2_path, index=False)
    print(f"\n表2(カイ二乗検定) -> {table2_path}")
    print(chi2_out.to_string(index=False))

    # ---- LaTeX出力(表1のみ、ポスター貼り付け用) ----
    tex_path = TABLES_DIR / "corrected_poster_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(table1.to_latex(index=False, escape=True,
                                 caption="Corrected catalog/beef/post conflict-vocabulary and "
                                         "opponent-mention rates (real v2 data)",
                                 label="tab:corrected_poster_table"))
    print(f"\nLaTeX -> {tex_path}")


if __name__ == "__main__":
    main()
