"""
plot_timeseries.py

analysis/ 配下の各スクリプトが生成したCSVを読み込み、ポスター発表用の図をPNGで出力する。

出力先: beef_analysis/figures/
  conflict_vocab_timeseries.png   PRE/PEAK/POSTの対立語彙率・相手言及率（beef1/2/3）
  engagement_per_view.png         期間別の comment_per_view / like_per_view（beef1/2/3）
  supplement_persistence.png      補助分析（beef4/5）の対立語彙率・相手言及率
  {beef_folder}_lsa_clusters.png  期間別のLSA散布図（beef毎）

前提: conflict_vocab.py / engagement_metrics.py / tfidf_lsa.py を先に実行しておくこと。

使い方:
  python plot_timeseries.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BEEF_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "figures"

# 検証済みカテゴリカルパレット（固定順）。beef1->slot1, beef2->slot2, beef3->slot3
CATEGORICAL = {
    "beef1": "#2a78d6",  # blue
    "beef2": "#1baf7a",  # aqua
    "beef3": "#eda100",  # yellow
    "beef4": "#4a3aa7",  # violet
    "beef5": "#e34948",  # red
}
BEEF_LABELS = {
    "beef1": "Beef1: Kendrick vs Drake",
    "beef2": "Beef2: Pusha T vs Drake",
    "beef3": "Beef3: Wayne vs Birdman",
    "beef4": "Beef4: Jay-Z vs Nas",
    "beef5": "Beef5: 2Pac vs Biggie",
}
PERIOD_ORDER = ["PRE", "PEAK", "POST"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_COLOR)
    ax.spines["bottom"].set_color(BASELINE_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_conflict_vocab_timeseries():
    path = DATA_PROCESSED / "conflict_vocab_summary.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（スキップ）。先に conflict_vocab.py を実行してください。")
        return

    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    main = df[df["beef_id"].isin(["beef1", "beef2", "beef3"])]
    if main.empty:
        print("conflict_vocab_timeseries: main beef（beef1/2/3）のデータが0件です。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    metrics = [("pct_conflict_vocab", "Conflict vocabulary rate (%)"),
               ("pct_opponent_mention", "Opponent-name mention rate (%)")]

    for ax, (metric, title) in zip(axes, metrics):
        for beef_id in ["beef1", "beef2", "beef3"]:
            sub = main[main["beef_id"] == beef_id].set_index("period").reindex(PERIOD_ORDER)
            ax.plot(PERIOD_ORDER, sub[metric], color=CATEGORICAL[beef_id], linewidth=2,
                    marker="o", markersize=8)
            last_valid = sub[metric].dropna()
            if not last_valid.empty:
                ax.annotate(BEEF_LABELS[beef_id], xy=(2, last_valid.iloc[-1]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=CATEGORICAL[beef_id], va="center")
        ax.set_title(title, fontsize=11, color=INK_PRIMARY, loc="left")
        ax.set_xlim(-0.3, 2.9)
        style_axis(ax)

    fig.suptitle("Conflict discourse over time (PRE -> PEAK -> POST)", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "conflict_vocab_timeseries.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_engagement_per_view():
    path = DATA_PROCESSED / "engagement_by_video.csv"
    songs_beef_map = DATA_PROCESSED.parent / "config" / "songs.csv"
    viewcount_path = DATA_PROCESSED / "viewcount_by_period.csv"
    if not path.exists() or not viewcount_path.exists():
        print("警告: engagement_by_video.csv / viewcount_by_period.csv が見つかりません（スキップ）。"
              "先に engagement_metrics.py / viewcount_analysis.py を実行してください。")
        return

    engagement = pd.read_csv(path)
    release_period = pd.read_csv(
        viewcount_path, keep_default_na=False, na_values=[""]
    )[["beef_id", "song_name", "release_period"]]
    merged = pd.merge(engagement, release_period, on=["beef_id", "song_name"], how="left")
    main = merged[merged["beef_id"].isin(["beef1", "beef2", "beef3"])]
    main = main[main["release_period"].isin(PERIOD_ORDER)]
    if main.empty:
        print("engagement_per_view: main beefのデータが0件です。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    metrics = [("comment_per_view", "Comments per view"), ("like_per_view", "Likes per view")]
    bar_width = 0.25

    for ax, (metric, title) in zip(axes, metrics):
        for i, beef_id in enumerate(["beef1", "beef2", "beef3"]):
            sub = main[main["beef_id"] == beef_id].groupby("release_period")[metric].mean().reindex(PERIOD_ORDER)
            positions = [x + i * bar_width for x in range(len(PERIOD_ORDER))]
            ax.bar(positions, sub.values, width=bar_width, color=CATEGORICAL[beef_id],
                   label=BEEF_LABELS[beef_id])
        ax.set_xticks([x + bar_width for x in range(len(PERIOD_ORDER))])
        ax.set_xticklabels(PERIOD_ORDER)
        ax.set_title(title, fontsize=11, color=INK_PRIMARY, loc="left")
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-view engagement by release period", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "engagement_per_view.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_supplement_persistence():
    path = DATA_PROCESSED / "conflict_vocab_summary.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（スキップ）。")
        return

    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    supplement = df[df["beef_id"].isin(["beef4", "beef5"])]
    if supplement.empty:
        print("supplement_persistence: beef4/5のデータが0件です。")
        return

    fig, ax = plt.subplots(figsize=(6, 4.2))
    metrics = ["pct_conflict_vocab", "pct_opponent_mention"]
    bar_width = 0.35
    x = range(len(metrics))

    for i, beef_id in enumerate(["beef4", "beef5"]):
        sub = supplement[supplement["beef_id"] == beef_id]
        values = [sub[m].mean() for m in metrics]
        positions = [xi + i * bar_width for xi in x]
        ax.bar(positions, values, width=bar_width, color=CATEGORICAL[beef_id], label=BEEF_LABELS[beef_id])

    ax.set_xticks([xi + bar_width / 2 for xi in x])
    ax.set_xticklabels(["Conflict vocabulary rate (%)", "Opponent-name mention rate (%)"], fontsize=9)
    ax.set_title("Historical beefs: present-day discourse persistence", fontsize=12, color=INK_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axis(ax)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "supplement_persistence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_lsa_clusters():
    for beef_folder in ["beef1_kendrick_drake", "beef2_pusha_drake", "beef3_wayne_birdman"]:
        path = DATA_PROCESSED / f"{beef_folder}_lsa_projection.csv"
        if not path.exists():
            print(f"警告: {path} が見つかりません（スキップ）。先に tfidf_lsa.py を実行してください。")
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        beef_id = beef_folder.split("_")[0]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        period_colors = {"PRE": "#2a78d6", "PEAK": "#1baf7a", "POST": "#eda100"}
        for period in PERIOD_ORDER:
            sub = df[df["period"] == period]
            if sub.empty:
                continue
            ax.scatter(sub["pc1"], sub["pc2"], s=10, alpha=0.5, color=period_colors[period], label=period)

        ax.set_title(f"{BEEF_LABELS.get(beef_id, beef_id)}: LSA comment clusters", fontsize=11, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("Component 1", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Component 2", fontsize=9, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=9)
        style_axis(ax)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"{beef_folder}_lsa_clusters.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"保存: {out_path}")


def main():
    plot_conflict_vocab_timeseries()
    plot_engagement_per_view()
    plot_supplement_persistence()
    plot_lsa_clusters()


if __name__ == "__main__":
    main()
