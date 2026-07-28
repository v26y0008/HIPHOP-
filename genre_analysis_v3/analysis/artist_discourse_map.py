"""
artist_discourse_map.py (genre_v3)

lyric_axis_beef.py で計算した lyric_rate / vibe_rate を使い、
アーティスト x track_category を1点とした2次元散布図（lyric_rate vs vibe_rate）を描く。
色はgenre（boom_bap/trap/alternative/beef_context）、形はtrack_category。

lyric_vibe_per_artist_category.csv（lyric_axis_beef.py の出力）が必要。

使い方:
  python artist_discourse_map.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"

GENRE_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = GENRE_ROOT / "data" / "processed" / "lyric_vibe_per_artist_category.csv"
FIGURES_DIR = GENRE_ROOT / "outputs" / "figures"

GENRE_COLORS = {
    "boom_bap": "#0072B2", "trap": "#E69F00", "alternative": "#009E73", "beef_context": "#999999",
}
CATEGORY_MARKERS = {"catalog": "o", "beef": "^", "post": "s"}

MIN_N = 20  # サンプル数が少なすぎる artist x category は除外（点の信頼性が低いため）


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に lyric_axis_beef.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df = df[df["n"] >= MIN_N].copy()
    print(f"プロット対象: {len(df)}点（n>={MIN_N}件でフィルタ）")

    fig, ax = plt.subplots(figsize=(9, 7.5))

    for genre, color in GENRE_COLORS.items():
        sub = df[df["genre"] == genre]
        for cat, marker in CATEGORY_MARKERS.items():
            sub_cat = sub[sub["track_category"] == cat]
            if len(sub_cat) == 0:
                continue
            label = f"{genre} / {cat}"
            ax.scatter(sub_cat["lyric_rate"], sub_cat["vibe_rate"],
                       c=color, marker=marker, s=70, alpha=0.8,
                       edgecolors="white", linewidths=0.5, label=label)

    for _, row in df.iterrows():
        label = row["artist_name"].replace("$", "\\$")
        ax.annotate(label, (row["lyric_rate"], row["vibe_rate"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Lyric-vocabulary rate in comments (%)")
    ax.set_ylabel("Vibe-vocabulary rate in comments (%)")
    ax.set_title("Artist discourse map: lyric-focused vs vibe-focused comments\n(marker=track_category, color=genre)")

    # 凡例はgenreとcategoryをそれぞれ簡潔に
    from matplotlib.lines import Line2D
    genre_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=g)
                      for g, c in GENRE_COLORS.items()]
    cat_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="", markersize=9, label=c)
                   for c, m in CATEGORY_MARKERS.items()]
    legend1 = ax.legend(handles=genre_handles, title="genre", loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=cat_handles, title="track_category", loc="upper right", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_genre_v3_artist_discourse_map.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure保存: {out_path}")


if __name__ == "__main__":
    main()
