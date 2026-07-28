"""
generate_tables.py (v2 設計)

ポスター・論文用の結果表を CSV と LaTeX の両形式で outputs/tables/ に出力する。

Table 1: ビーフ×track_category別コメント件数と主要指標
Table 2: カイ二乗検定結果（Bonferroni補正込み）
Table 3: comment_per_view ランキング上位10動画

使い方:
  python generate_tables.py
"""

from pathlib import Path

import pandas as pd

BEEF_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"
STATS_DIR = BEEF_ROOT / "outputs" / "stats"
TABLES_DIR = BEEF_ROOT / "outputs" / "tables"

BEEF_LABELS = {
    "b1": "B1: Kendrick vs Drake", "b2": "B2: Pusha T vs Drake",
    "b4": "B4: Eminem vs MGK", "b5": "B5: Wayne vs Birdman", "b6": "B6: Megan vs Nicki",
}


def save_table(df, name, caption):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=True, caption=caption, label=f"tab:{name}"))
    print(f"{name}: -> {csv_path}, {tex_path}")


def table1_summary():
    path = DATA_PROCESSED / "summary_by_beef_category.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（Table1をスキップ）")
        return
    df = pd.read_csv(path)
    df = df[["beef_id", "track_category", "n_comments", "pct_conflict", "pct_opponent", "avg_length"]]
    df["beef_id"] = df["beef_id"].map(lambda b: BEEF_LABELS.get(b, b))
    save_table(df, "table1_summary", "Comment counts and key metrics by beef and track category")


def table2_chi2_results():
    path = STATS_DIR / "chi2_results_v2.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（Table2をスキップ）")
        return
    df = pd.read_csv(path)
    df["beef_label"] = df["beef_id"].map(lambda b: BEEF_LABELS.get(b, b))
    df["sig_bonferroni"] = df.apply(
        lambda r: f"Yes (alpha={r['alpha_corrected']:.4f})" if r["significant_bonferroni"] else "No", axis=1
    )
    out = df[["beef_label", "beef_type", "target", "chi2", "p_value", "cramers_v", "sig_bonferroni"]].rename(
        columns={
            "beef_label": "Beef", "beef_type": "Beef type", "target": "Test target",
            "chi2": "chi2", "p_value": "p-value", "cramers_v": "Cramer's V",
            "sig_bonferroni": "Significant (Bonferroni)",
        }
    )
    save_table(out, "table2_chi2_results", "Chi-square test results across catalog/beef/post categories")


def table3_cpv_ranking():
    path = DATA_PROCESSED / "video_level_analysis_v2.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（Table3をスキップ）")
        return
    df = pd.read_csv(path).sort_values("comment_per_view", ascending=False).head(10).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    out = df[["rank", "song_name", "artist_name", "track_category", "comment_per_view"]].rename(
        columns={
            "rank": "Rank", "song_name": "Song", "artist_name": "Artist",
            "track_category": "Category", "comment_per_view": "Comments per view",
        }
    )
    save_table(out, "table3_comment_per_view", "Top 10 videos by comments per view")


def main():
    table1_summary()
    table2_chi2_results()
    table3_cpv_ranking()


if __name__ == "__main__":
    main()
