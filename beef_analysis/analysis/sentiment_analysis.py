"""
sentiment_analysis.py (v2 追加モジュール)

対立語彙率の分析を補強するため、VADER(ルールベース感情分析)で
Positive/Negative/Neutralの分布を追加する。

分析の軸:
  1. beef_id x track_category(catalog/beef/post) x sentiment の分布
     → 既存の「対立語彙率はbeef曲で上昇」という結果と整合するか
  2. 勝者側 vs 敗者側(beefカテゴリのみ)の感情分布比較
     → 既存の「敗者側の対立語彙率が高い」結果と、感情面でも整合するか

使い方:
  python sentiment_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

matplotlib.rcParams["font.family"] = "Meiryo"

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
WIN_LOSS_PATH = BEEF_ROOT / "data" / "processed" / "comments_with_win_loss_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

CATEGORY_ORDER = ["catalog", "beef", "post"]
SENTIMENT_COLORS = {"positive": "#1A9E6A", "neutral": "#999999", "negative": "#CC4400"}

BEEF_LABELS = {
    "b1": "B1: Kendrick vs Drake", "b2": "B2: Pusha T vs Drake",
    "b4": "B4: Eminem vs MGK", "b5": "B5: Wayne vs Birdman", "b6": "B6: Megan vs Nicki",
}

analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    print(f"対象コメント数: {len(df):,}")

    print("VADER感情分析を実行中...")
    df["sentiment"] = df["text_clean"].apply(get_sentiment)
    print("完了")

    df.to_csv(OUT_DIR / "comments_with_sentiment_v2.csv", index=False)

    # === 1) beef_id x track_category x sentiment ===
    summary = df.groupby(["beef_id", "track_category", "sentiment"]).size().unstack(fill_value=0)
    summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100
    for s in ["positive", "neutral", "negative"]:
        if s not in summary_pct.columns:
            summary_pct[s] = 0.0
    summary_pct = summary_pct[["positive", "neutral", "negative"]]
    print("\n=== beef_id x track_category別 感情分布(%) ===")
    print(summary_pct.round(1))
    summary_pct.to_csv(OUT_DIR / "sentiment_by_beef_phase.csv")

    # カテゴリ×感情のカイ二乗検定(beef_id別)
    print("\n=== track_category x sentiment カイ二乗検定(beef_idごと) ===")
    chi2_rows = []
    for bid in df["beef_id"].unique():
        sub = df[df["beef_id"] == bid]
        ct = pd.crosstab(sub["track_category"], sub["sentiment"])
        ct = ct.reindex(CATEGORY_ORDER).dropna()
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.values.sum()
        v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
        chi2_rows.append({"beef_id": bid, "chi2": chi2, "p_value": p, "cramers_v": v, "n": n})
        print(f"  {bid}: chi2={chi2:.2f}, p={p:.4g}, Cramer's V={v:.3f}, n={n}")
    pd.DataFrame(chi2_rows).to_csv(OUT_DIR / "sentiment_chi2_by_beef.csv", index=False)

    # === 図: beef_id x track_category 積み上げ棒グラフ ===
    beef_ids = [b for b in ["b1", "b2", "b4", "b5", "b6"] if b in summary_pct.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(beef_ids), figsize=(3.6 * len(beef_ids), 5), sharey=True)
    if len(beef_ids) == 1:
        axes = [axes]
    for ax, bid in zip(axes, beef_ids):
        sub = summary_pct.loc[bid].reindex(CATEGORY_ORDER).dropna()
        bottom = pd.Series(0.0, index=sub.index)
        for s in ["negative", "neutral", "positive"]:
            ax.bar(sub.index, sub[s], bottom=bottom, color=SENTIMENT_COLORS[s], label=s, edgecolor="white")
            bottom += sub[s]
        ax.set_title(BEEF_LABELS.get(bid, bid), fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("割合 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=10)
    fig.suptitle("感情分布 (VADER): track_category別", y=1.1, fontsize=13)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_sentiment_by_phase.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure保存: {out_path}")

    # === 2) 勝者側 vs 敗者側(beefカテゴリ) ===
    wl = pd.read_csv(WIN_LOSS_PATH, keep_default_na=False, na_values=[""])
    wl["sentiment"] = wl["text_clean"].apply(get_sentiment)
    beef_wl = wl[(wl["track_category"] == "beef") & (wl["win_loss_side"].isin(["winner", "loser"]))]

    wl_summary = beef_wl.groupby(["win_loss_side", "sentiment"]).size().unstack(fill_value=0)
    wl_summary_pct = wl_summary.div(wl_summary.sum(axis=1), axis=0) * 100
    for s in ["positive", "neutral", "negative"]:
        if s not in wl_summary_pct.columns:
            wl_summary_pct[s] = 0.0
    wl_summary_pct = wl_summary_pct[["positive", "neutral", "negative"]]
    print("\n=== 勝者側 vs 敗者側(beefカテゴリ) 感情分布(%) ===")
    print(wl_summary_pct.round(1))
    wl_summary_pct.to_csv(OUT_DIR / "sentiment_by_win_loss.csv")

    ct_wl = pd.crosstab(beef_wl["win_loss_side"], beef_wl["sentiment"])
    chi2_wl, p_wl, dof_wl, _ = chi2_contingency(ct_wl)
    n_wl = ct_wl.values.sum()
    v_wl = (chi2_wl / (n_wl * (min(ct_wl.shape) - 1))) ** 0.5
    print(f"\n勝者 vs 敗者 x sentiment カイ二乗検定: chi2={chi2_wl:.2f}, p={p_wl:.4g}, "
          f"Cramer's V={v_wl:.3f}, n={n_wl}")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    order = ["winner", "loser"]
    bottom = pd.Series(0.0, index=order)
    for s in ["negative", "neutral", "positive"]:
        vals = wl_summary_pct.reindex(order)[s]
        ax2.bar(order, vals, bottom=bottom, color=SENTIMENT_COLORS[s], label=s, edgecolor="white")
        bottom += vals
    ax2.set_ylabel("割合 (%)")
    ax2.set_title(f"感情分布: 勝者側 vs 敗者側 (beef曲, n={n_wl:,})\n"
                  f"chi2={chi2_wl:.1f}, p={p_wl:.2g}, V={v_wl:.3f}")
    ax2.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    out_path2 = FIGURES_DIR / "fig_sentiment_winner_loser.png"
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure保存: {out_path2}")


if __name__ == "__main__":
    main()
