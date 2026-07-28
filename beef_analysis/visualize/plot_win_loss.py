"""
plot_win_loss.py (v2 追加モジュール)

勝敗分析の可視化（ポスター用、DPI300、Okabe-Itoベース）。

Figure: 勝者側 vs 敗者側のビーフ曲対立語彙率比較（ビーフ別）
Figure: カタログ→ビーフ曲→POSTの勝者・敗者別comment_per_view推移

使い方:
  python plot_win_loss.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"

BEEF_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"
DPI = 300

COLORS = {"winner": "#1A9E6A", "loser": "#D4820A"}

BEEF_OUTCOME = {
    "b1": {"winner": "kendrick", "loser": "drake", "verdict": "clear"},
    "b2": {"winner": "pusha", "loser": "drake", "verdict": "clear"},
    "b4": {"winner": "eminem", "loser": "mgk", "verdict": "clear"},
    "b5": {"winner": "wayne", "loser": "birdman", "verdict": "clear"},
    "b6": {"winner": "megan", "loser": "nicki", "verdict": "disputed"},
}
BEEF_NAMES = {
    "b1": "Kendrick vs Drake", "b2": "Pusha T vs Drake", "b4": "Eminem vs MGK",
    "b5": "Wayne vs Birdman", "b6": "Megan vs Nicki*",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_COLOR)
    ax.spines["bottom"].set_color(BASELINE_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_win_loss_vocab(df):
    beef_df = df[df["track_category"] == "beef"]
    beef_ids = sorted(beef_df["beef_id"].unique())

    fig, axes = plt.subplots(1, len(beef_ids), figsize=(3.2 * len(beef_ids), 4.5), sharey=True)
    for ax, bid in zip(axes, beef_ids):
        outcome = BEEF_OUTCOME.get(bid, {})
        w_side, l_side = outcome.get("winner", ""), outcome.get("loser", "")

        w_sub = beef_df[(beef_df["beef_id"] == bid) & (beef_df["artist_side"] == w_side)]
        l_sub = beef_df[(beef_df["beef_id"] == bid) & (beef_df["artist_side"] == l_side)]
        w_pct = w_sub["has_conflict"].mean() * 100 if len(w_sub) else None
        l_pct = l_sub["has_conflict"].mean() * 100 if len(l_sub) else None

        bars = ax.bar(["Winner", "Loser"], [w_pct or 0, l_pct or 0],
                       color=[COLORS["winner"], COLORS["loser"]], width=0.5)
        for bar, val in zip(bars, [w_pct, l_pct]):
            if val is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.5, "no data",
                        ha="center", fontsize=8, color=INK_SECONDARY, rotation=90, va="bottom")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%",
                        ha="center", fontsize=9, color=INK_SECONDARY)
        title = BEEF_NAMES.get(bid, bid)
        if outcome.get("verdict") == "disputed":
            title += "\n(disputed)"
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        style_axis(ax)
    axes[0].set_ylabel("Conflict vocabulary rate in beef tracks (%)", fontsize=10, color=INK_SECONDARY)
    fig.suptitle("Winner-side vs. loser-side fans: conflict vocabulary in beef-track comments",
                 fontsize=13, color=INK_PRIMARY)
    fig.text(0.01, 0.01,
              "* B6 (Megan vs Nicki) outcome is disputed in media coverage; direction holds without it too.\n"
              "\"no data\" = comments disabled on that artist's beef video (B1 loser/Push Ups, "
              "B2 winner/Story of Adidon) or no beef track exists (B5 loser/Birdman) - not a true 0%.",
              fontsize=7, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig5_win_loss_conflict_vocab_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_post_track_comparison(df):
    metadata_path = DATA_PROCESSED / "all_video_metadata_v2.csv"
    if not metadata_path.exists():
        print(f"警告: {metadata_path} が見つかりません（スキップ）")
        return
    video_stats = pd.read_csv(metadata_path)[["video_id", "view_count"]]

    video_agg = df.groupby(["video_id", "beef_id", "track_category", "artist_side", "win_loss_side"]).agg(
        n_comments=("comment_id", "count"),
    ).reset_index()
    video_agg = video_agg.merge(video_stats, on="video_id")
    video_agg["comment_per_view"] = video_agg["n_comments"] / video_agg["view_count"].clip(lower=1)

    beef_ids = sorted(BEEF_OUTCOME.keys())
    fig, axes = plt.subplots(1, len(beef_ids), figsize=(3.4 * len(beef_ids), 4.5), sharey=False)
    cats = ["catalog", "beef", "post"]

    for ax, bid in zip(axes, beef_ids):
        b_df = video_agg[video_agg["beef_id"] == bid]
        for side, color, label in [("winner", COLORS["winner"], "Winner"), ("loser", COLORS["loser"], "Loser")]:
            side_df = b_df[b_df["win_loss_side"] == side]
            vals = [side_df[side_df["track_category"] == c]["comment_per_view"].mean() for c in cats]
            ax.plot(["Catalog", "Beef", "Post"], vals, marker="o", color=color, label=label, linewidth=2)
        ax.set_title(BEEF_NAMES.get(bid, bid), fontsize=9, color=INK_PRIMARY)
        ax.set_ylabel("Comments per view", fontsize=8, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=8)
        style_axis(ax)

    fig.suptitle("Winner vs. loser: comment intensity from catalog to beef to post\n"
                 "(each panel own y-scale - not for cross-beef magnitude comparison, see fig4 for that)",
                 fontsize=12, color=INK_PRIMARY)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = FIGURES_DIR / "fig6_win_loss_post_cpv_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def main():
    path = DATA_PROCESSED / "comments_with_win_loss_v2.csv"
    if not path.exists():
        print(f"エラー: {path} が見つかりません。先に win_loss_analysis.py を実行してください。")
        return
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    plot_win_loss_vocab(df)
    plot_post_track_comparison(df)


if __name__ == "__main__":
    main()
