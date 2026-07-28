"""
plot_poster.py (v2 設計)

track_category（catalog/beef/post）設計に基づくポスター用の図をDPI300で出力する。
色はOkabe-Itoパレット（色覚多様性対応）。

Figure 1: ビーフ別・カテゴリ別 対立語彙出現率
Figure 2: カテゴリ別 comment_per_view（カタログ vs ビーフ vs ポスト）
Figure 3: ビーフタイプ別（rap_battle vs contract_breakdown）対立語彙率比較
Figure 4: comment_per_view ランキング（動画別）

使い方:
  python plot_poster.py
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

COLORS = {
    "catalog": "#999999",
    "beef": "#0072B2",
    "post": "#E69F00",
    "b1": "#0072B2", "b2": "#009E73", "b4": "#D55E00", "b5": "#F0E442", "b6": "#56B4E9",
}
BEEF_LABELS = {
    "b1": "B1: Kendrick vs Drake", "b2": "B2: Pusha T vs Drake",
    "b4": "B4: Eminem vs MGK", "b5": "B5: Wayne vs Birdman", "b6": "B6: Megan vs Nicki",
}
CATEGORY_ORDER = ["catalog", "beef", "post"]
CATEGORY_LABELS_EN = {"catalog": "Catalog", "beef": "Beef", "post": "Post"}

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


def plot_fig1_conflict_by_beef_category():
    path = DATA_PROCESSED / "summary_by_beef_category.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（スキップ）")
        return
    df = pd.read_csv(path)
    beef_ids = sorted(df["beef_id"].unique())

    fig, axes = plt.subplots(1, len(beef_ids), figsize=(3.2 * len(beef_ids), 4.5), sharey=True)
    for ax, bid in zip(axes, beef_ids):
        df_b = df[df["beef_id"] == bid].set_index("track_category").reindex(CATEGORY_ORDER)
        bars = ax.bar(
            [CATEGORY_LABELS_EN[c] for c in CATEGORY_ORDER], df_b["pct_conflict"],
            color=[COLORS[c] for c in CATEGORY_ORDER],
        )
        for bar, val in zip(bars, df_b["pct_conflict"]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%",
                        ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_title(BEEF_LABELS.get(bid, bid), fontsize=10, color=INK_PRIMARY)
        style_axis(ax)
    axes[0].set_ylabel("Conflict vocabulary rate (%)", fontsize=10, color=INK_SECONDARY)
    fig.suptitle("Conflict vocabulary rate: catalog vs. beef vs. post tracks", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig1_conflict_vocab_by_beef_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_fig2_comment_per_view_by_category():
    path = DATA_PROCESSED / "video_level_analysis_v2.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（スキップ）")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = df.groupby("track_category")["comment_per_view"].mean().reindex(CATEGORY_ORDER)
    bars = ax.bar([CATEGORY_LABELS_EN[c] for c in CATEGORY_ORDER], means,
                  color=[COLORS[c] for c in CATEGORY_ORDER])
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.5f}",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Comments per view (mean across videos)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Comment intensity by track category", fontsize=12, color=INK_PRIMARY, loc="left")
    style_axis(ax)
    fig.text(0.01, 0.01,
              "Note: regression controlling for days_since_release finds a positive but\n"
              "non-significant beef-track effect (p=0.16, n=26 videos) — directional only.",
              fontsize=7, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = FIGURES_DIR / "fig2_comment_per_view_by_category_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_fig3_beef_type_comparison():
    path = DATA_PROCESSED / "summary_by_beef_category.csv"
    chi2_path = BEEF_ROOT / "outputs" / "stats" / "chi2_results_v2.csv"
    if not path.exists() or not chi2_path.exists():
        print("警告: 必要なCSVが見つかりません（スキップ: fig3）")
        return
    df = pd.read_csv(path)
    chi2_df = pd.read_csv(chi2_path)

    beef_type_map = chi2_df.drop_duplicates("beef_id").set_index("beef_id")["beef_type"].to_dict()
    df["beef_type"] = df["beef_id"].map(beef_type_map)
    beef_only = df[df["track_category"] == "beef"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    type_colors = {"rap_battle": "#0072B2", "contract_breakdown": "#E69F00"}
    for i, bid in enumerate(sorted(beef_only["beef_id"].unique())):
        row = beef_only[beef_only["beef_id"] == bid].iloc[0]
        color = type_colors.get(row["beef_type"], "#999999")
        ax.bar(i, row["pct_conflict"], color=color)
        ax.text(i, row["pct_conflict"] + 0.3, f"{row['pct_conflict']:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(beef_only)))
    ax.set_xticklabels([BEEF_LABELS.get(b, b).split(":")[0] for b in sorted(beef_only["beef_id"].unique())])
    ax.set_ylabel("Conflict vocabulary rate in beef tracks (%)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Rap-battle beefs vs. contract-breakdown beefs", fontsize=12, color=INK_PRIMARY, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in type_colors.values()]
    ax.legend(handles, ["Rap battle", "Contract breakdown"], frameon=False, fontsize=9)
    style_axis(ax)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig3_beef_type_comparison_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_fig4_cpv_ranking():
    path = DATA_PROCESSED / "video_level_analysis_v2.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（スキップ）")
        return
    df = pd.read_csv(path).sort_values("comment_per_view")
    df["label"] = df["song_name"] + " (" + df["artist_name"] + ")"

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1.5))
    colors = df["track_category"].map(COLORS)
    ax.barh(df["label"], df["comment_per_view"], color=colors)
    ax.set_xlabel("Comments per view", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Comment intensity ranking (all videos)", fontsize=13, color=INK_PRIMARY, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CATEGORY_ORDER]
    ax.legend(handles, [CATEGORY_LABELS_EN[c] for c in CATEGORY_ORDER], frameon=False, fontsize=9, loc="lower right")
    style_axis(ax)
    ax.yaxis.grid(False)
    fig.tight_layout()

    out_path = FIGURES_DIR / "fig4_cpv_ranking_v2.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def main():
    plot_fig1_conflict_by_beef_category()
    plot_fig2_comment_per_view_by_category()
    plot_fig3_beef_type_comparison()
    plot_fig4_cpv_ranking()


if __name__ == "__main__":
    main()
