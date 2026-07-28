"""
plot_viewcount.py (v2 追加モジュール)

再生数分析の可視化（ポスター用 DPI300、Okabe-Itoベース）。

Figure 7: アーティスト別 カタログ→POST の再生効率推移（勝者=緑、敗者=橙）
Figure 8: 勝者 vs 敗者 の再生効率変化率の分布（箱ひげ図）

使い方:
  python plot_viewcount.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"

BEEF_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"
DPI = 300

COLORS = {"winner": "#1A9E6A", "loser": "#D4820A"}
BEEF_NAMES = {
    "b1": "Kendrick vs Drake", "b2": "Pusha T vs Drake", "b4": "Eminem vs MGK",
    "b5": "Wayne vs Birdman", "b6": "Megan vs Nicki*",
}
BEEF_OUTCOME = {
    "b1": {"winner": "kendrick", "loser": "drake"},
    "b2": {"winner": "pusha", "loser": "drake"},
    "b4": {"winner": "eminem", "loser": "mgk"},
    "b5": {"winner": "wayne", "loser": "birdman"},
    "b6": {"winner": "megan", "loser": "nicki"},
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
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_efficiency_by_artist(df):
    beef_ids = sorted(BEEF_OUTCOME.keys())
    fig, axes = plt.subplots(1, len(beef_ids), figsize=(3.4 * len(beef_ids), 4.5), sharey=False)
    cats = ["catalog", "beef", "post"]

    for ax, bid in zip(axes, beef_ids):
        outcome = BEEF_OUTCOME[bid]
        b_df = df[df["beef_id"] == bid]
        for wl, side, color in [("winner", outcome["winner"], COLORS["winner"]),
                                 ("loser", outcome["loser"], COLORS["loser"])]:
            vals = []
            for cat in cats:
                sub = b_df[(b_df["artist_side"] == side) & (b_df["track_category"] == cat)]["view_efficiency"]
                vals.append(sub.median() if len(sub) > 0 else np.nan)
            label = f"{'Winner' if wl == 'winner' else 'Loser'} ({side})"
            ax.plot(["Catalog", "Beef", "Post"], vals, marker="o", color=color, label=label, linewidth=2)
        ax.set_title(BEEF_NAMES.get(bid, bid), fontsize=9, color=INK_PRIMARY)
        ax.set_ylabel("Views per day since release", fontsize=8, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=7)
        style_axis(ax)

    fig.suptitle("View efficiency: catalog (recency-matched) -> beef -> post, by winner/loser",
                 fontsize=12, color=INK_PRIMARY)
    fig.text(0.01, 0.01,
              "* B6 (Megan vs Nicki) outcome disputed. Catalog picks are recency-matched "
              "(top-viewed within ~1-3yr before the beef, not all-time hits) to avoid an age bias.",
              fontsize=7, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig7_view_efficiency_by_artist_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_change_distribution(results_df):
    winner_changes = results_df[results_df["win_loss"] == "勝者"]["change_pct"]
    loser_changes = results_df[results_df["win_loss"] == "敗者"]["change_pct"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [winner_changes.dropna(), loser_changes.dropna()],
        tick_labels=["Winner", "Loser"], patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor(COLORS["winner"])
    bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][1].set_facecolor(COLORS["loser"])
    bp["boxes"][1].set_alpha(0.55)

    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1)
    ax.set_ylabel("Post-beef change in view efficiency (%)\nvs. recency-matched catalog", fontsize=9, color=INK_SECONDARY)
    ax.set_title("Post-beef view-efficiency change: winner vs. loser", fontsize=12, color=INK_PRIMARY, loc="left")
    style_axis(ax)
    fig.text(0.02, 0.02,
              "Mann-Whitney U (winner > loser, one-sided): p=0.65 (all 5 beefs), "
              "p=0.76 (clear-verdict only) - not significant.",
              fontsize=7, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = FIGURES_DIR / "fig8_view_efficiency_change_dist_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def main():
    eff_path = DATA_PROCESSED / "video_stats_with_efficiency_v2.csv"
    results_path = DATA_PROCESSED / "viewcount_results_v2.csv"
    if not eff_path.exists() or not results_path.exists():
        print("エラー: 先に viewcount_analysis.py を実行してください。")
        return
    df = pd.read_csv(eff_path)
    results_df = pd.read_csv(results_path)
    plot_efficiency_by_artist(df)
    plot_change_distribution(results_df)


if __name__ == "__main__":
    main()
