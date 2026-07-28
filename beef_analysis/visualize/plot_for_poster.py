"""
plot_for_poster.py

学会ポスター用の4図を生成し、outputs/figures/ にDPI300で保存する。

Figure 1: 対立語彙率の時系列変化（PRE/PEAK/POST、Beef1/2/3）
Figure 2: 相手アーティスト名言及率の時系列変化
Figure 3: comment_per_view の比較（ビーフ関連曲 vs カタログ曲）
Figure 4: Beef1/2/3 PEAK期の上位語彙ヒートマップ

前提: cross_beef_comparison.py, tfidf_lsa.py, analysis/preprocess.py 等を
先に実行し、data/processed/ 配下のCSVが揃っていること。

使い方:
  python plot_for_poster.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.rcParams["font.family"] = "sans-serif"

BEEF_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"
CONFIG_DIR = BEEF_ROOT / "config"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"
DPI = 300

# Okabe-Ito（色覚多様性対応）
COLORS = {
    "beef1": "#0072B2",       # 青
    "beef2": "#009E73",       # 緑
    "beef3": "#E69F00",       # 橙
    "supplement": "#999999",  # グレー
}
BEEF_LABELS = {
    "beef1": "Beef1: Kendrick vs Drake",
    "beef2": "Beef2: Pusha T vs Drake",
    "beef3": "Beef3: Wayne vs Birdman",
}
PERIOD_ORDER = ["PRE", "PEAK", "POST"]

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


def plot_timeseries_figure(metric_col, ylabel, title, out_name, significance_note):
    path = DATA_PROCESSED / "cross_beef_period_summary.csv"
    if not path.exists():
        print(f"警告: {path} が見つかりません（スキップ: {out_name}）")
        return

    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    main = df[df["beef_id"].isin(["beef1", "beef2", "beef3"])]

    fig, ax = plt.subplots(figsize=(7, 5))
    for beef_id in ["beef1", "beef2", "beef3"]:
        sub = main[main["beef_id"] == beef_id].set_index("period").reindex(PERIOD_ORDER)
        linestyle = "--" if beef_id == "beef3" else "-"
        ax.plot(
            PERIOD_ORDER, sub[metric_col], color=COLORS[beef_id], linewidth=2,
            linestyle=linestyle, marker="o", markersize=8, label=BEEF_LABELS[beef_id],
        )

    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
    ax.set_title(title, fontsize=13, color=INK_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="best")
    style_axis(ax)
    if significance_note:
        fig.text(0.01, 0.01, significance_note, fontsize=7, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_fig1_conflict_vocab_timeseries():
    plot_timeseries_figure(
        "pct_conflict_vocab", "Conflict vocabulary rate (%)",
        "Conflict vocabulary over time (PRE -> PEAK -> POST)",
        "fig1_conflict_vocab_timeseries.png",
        "Beef3 (dashed) is not statistically significant (chi2 p=0.69); see test_results.txt",
    )


def plot_fig2_opponent_mention_timeseries():
    plot_timeseries_figure(
        "pct_opponent_mention", "Opponent-name mention rate (%)",
        "Opponent-name mentions over time (PRE -> PEAK -> POST)",
        "fig2_opponent_mention_timeseries.png",
        "All three beefs rise from ~0% at PRE to double digits at PEAK.",
    )


def plot_fig3_comment_per_view():
    metadata_path = DATA_PROCESSED / "all_video_metadata.csv"
    songs_path = CONFIG_DIR / "songs.csv"
    if not metadata_path.exists() or not songs_path.exists():
        print(f"警告: {metadata_path} または {songs_path} が見つかりません（スキップ: fig3）")
        return

    meta = pd.read_csv(metadata_path)
    songs = pd.read_csv(songs_path, keep_default_na=False, na_values=[""])[
        ["video_id", "role_in_beef"]
    ].drop_duplicates(subset="video_id")

    df = pd.merge(meta, songs, on="video_id", how="left")
    df = df.drop_duplicates(subset="video_id").copy()
    for col in ["view_count", "comment_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["view_count"] > 0]
    df["comment_per_view"] = df["comment_count"] / df["view_count"]
    df["category"] = np.where(df["role_in_beef"] == "PRE基準", "Catalog track", "Beef-related track")
    df["label"] = df["song_name"] + " (" + df["artist"] + ")"
    df = df.sort_values("comment_per_view", ascending=True)

    cat_colors = {"Beef-related track": COLORS["beef1"], "Catalog track": COLORS["supplement"]}
    bar_colors = df["category"].map(cat_colors)

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1.5))
    ax.barh(df["label"], df["comment_per_view"], color=bar_colors)
    ax.set_xlabel("Comments per view", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        "Comment intensity: beef-related tracks vs. catalog tracks",
        fontsize=13, color=INK_PRIMARY, loc="left",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_colors.values()]
    ax.legend(handles, cat_colors.keys(), frameon=False, fontsize=9, loc="lower right")
    style_axis(ax)
    ax.yaxis.grid(False)

    caption = (
        "Note: \"Not Like Us\" ranks mid-table despite the largest view count — its official\n"
        "YouTube upload lagged release by ~2 months (missing PRE/PEAK comments) and its\n"
        "reach extended beyond hip-hop fans into mainstream pop culture, diluting comments/view."
    )
    fig.text(0.01, 0.01, caption, fontsize=7, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig3_comment_per_view.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def _mean_tfidf_by_group(vectorizer, tfidf_matrix, group_masks):
    feature_names = np.array(vectorizer.get_feature_names_out())
    group_scores = {}
    for label, mask in group_masks.items():
        if mask.sum() == 0:
            group_scores[label] = np.zeros(len(feature_names))
        else:
            group_scores[label] = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
    return feature_names, group_scores


def plot_fig4_topic_heatmap():
    folders = {
        "beef1": "beef1_kendrick_drake",
        "beef2": "beef2_pusha_drake",
        "beef3": "beef3_wayne_birdman",
    }
    texts, groups = [], []
    for beef_id, folder in folders.items():
        path = DATA_PROCESSED / f"{folder}_comments_clean.csv"
        if not path.exists():
            print(f"警告: {path} が見つかりません（スキップ: fig4）")
            return
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        peak = df[df["period"] == "PEAK"].dropna(subset=["text_clean"])
        texts.append(peak["text_clean"])
        groups.append(beef_id)

    all_text = pd.concat(texts, ignore_index=True)
    boundaries = np.cumsum([len(t) for t in texts])
    group_masks = {}
    start = 0
    for beef_id, end in zip(groups, boundaries):
        mask = np.zeros(len(all_text), dtype=bool)
        mask[start:end] = True
        group_masks[beef_id] = mask
        start = end

    vectorizer = TfidfVectorizer(max_features=3000, min_df=3, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_text).toarray()
    feature_names, group_scores = _mean_tfidf_by_group(vectorizer, tfidf_matrix, group_masks)

    top_terms = []
    for beef_id in groups:
        top_idx = np.argsort(-group_scores[beef_id])[:8]
        for idx in top_idx:
            if feature_names[idx] not in top_terms:
                top_terms.append(feature_names[idx])
    top_terms = top_terms[:15]

    term_idx = {t: np.where(feature_names == t)[0][0] for t in top_terms}
    matrix = np.array([[group_scores[b][term_idx[t]] for b in groups] for t in top_terms])

    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(top_terms) + 1.8))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([BEEF_LABELS[b].split(":")[0] for b in groups], fontsize=9)
    ax.set_yticks(range(len(top_terms)))
    ax.set_yticklabels(top_terms, fontsize=9)
    fig.suptitle(
        "Top PEAK-period vocabulary:\nrap-battle framing vs. label-dispute framing",
        fontsize=12, color=INK_PRIMARY, x=0.02, ha="left",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean TF-IDF score", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig4_topic_heatmap.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"保存: {out_path}")


def main():
    plot_fig1_conflict_vocab_timeseries()
    plot_fig2_opponent_mention_timeseries()
    plot_fig3_comment_per_view()
    plot_fig4_topic_heatmap()


if __name__ == "__main__":
    main()
