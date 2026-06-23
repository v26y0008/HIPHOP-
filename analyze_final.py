"""
analyze_final.py
=================================================================
HIPHOPファン言説分析 ― 最終分析スクリプト

run_all_v5.py の出力（output_v5/results/ 内のCSV）を読み込み、
これまでの検証プロセスで確立した判定基準を自動的に適用して
最終レポートと図を生成する。

判定基準（これまでの手動検証で確立したルール）:
  1. 信頼度: 支配的トピックの件数が
       n>=50  -> ◎ 信頼できる
       10<=n<50 -> ▲ 要注意（サンプル数が少ない）
       n<10   -> ✗ 不十分
  2. 交絡スコア: そのトピックの言及が特定の1本の動画に
     集中していないか自動チェックする。
       最大1動画のシェア >= 70%  -> ⚠ 単一動画に集中（交絡の疑い）
       最大1動画のシェア < 50%   -> ✅ 複数動画に分散（交絡の疑い低い）
       その間                    -> △ 中間

使い方:
  python analyze_final.py
  (output_v5/results/ に必要なCSVがある前提)

出力:
  output_v5/results/final_summary.csv
  output_v5/results/figures/fig_final_summary.png
  output_v5/results/FINAL_REPORT.md
=================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

RESULT_DIR = "output_v5/results"
FIG_DIR = f"{RESULT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

TOPICS = ["⚔️ ビーフ", "🎪 ライブ", "🕯️ 追悼", "🔥 炎上"]

# これまでの独立した複数回の収集（v3/v4/v5）で再現性が確認されている
# アーティスト・トピックの組み合わせ。新しい結果がこれと一致するかの
# 参考情報として使う（このスクリプト単体では検証できないため）。
PREVIOUSLY_VALIDATED = {
    ("Mac Miller", "🕯️ 追悼"): "v3/v4/v5の3回の独立収集で8.3〜10.2%の範囲で再現済み（最も信頼できる発見）",
    ("Kendrick Lamar", "⚔️ ビーフ"): "v3/v4/v5の3回の独立収集で11.8〜30.3%の範囲で再現済み。ディス曲を除いても約17%残る",
    ("J. Cole", "⚔️ ビーフ"): "v3/v4の2回の独立収集で13.3〜17.8%の範囲で再現済み。ディス曲は1本もない",
    ("Travis Scott", "🎪 ライブ"): "v3/v4では20%前後だったが、MV限定で収集した結果0.7〜1.3%に低下。"
                                    "元の発見は『収集した動画がそもそもライブ/イベント映像だった』という交絡だったと判明（撤回・教訓として記録）",
    ("Tyler, The Creator", "🔥 炎上"): "v3で90.7%と報告したが、クラスタ単位判定の誤りと判明。"
                                        "実際の文字列マッチでは0.5%程度。撤回済み",
}


def load_data():
    counts = pd.read_csv(f"{RESULT_DIR}/artist_topic_counts.csv", index_col=0)
    pct = pd.read_csv(f"{RESULT_DIR}/artist_topic_pct.csv", index_col=0)
    confound = pd.read_csv(f"{RESULT_DIR}/confound_check_by_video.csv")
    return counts, pct, confound


def confidence_tier(n):
    if n >= 50:
        return "◎ 信頼できる"
    elif n >= 10:
        return "▲ 要注意（少数）"
    else:
        return "✗ 不十分"


def confound_tier(max_video_share):
    if max_video_share >= 0.70:
        return "⚠ 単一動画に集中"
    elif max_video_share < 0.50:
        return "✅ 複数動画に分散"
    else:
        return "△ 中間"


def compute_confound_scores(confound_df):
    """各アーティスト×トピックについて、最大1本の動画が占めるシェアを計算する"""
    scores = {}
    for (artist, topic), group in confound_df.groupby(["artist", "topic"]):
        total = group["count"].sum()
        if total == 0:
            continue
        max_video_count = group["count"].max()
        share = max_video_count / total
        n_videos_with_mentions = len(group)
        scores[(artist, topic)] = {
            "total_count": total,
            "max_video_share": round(share, 3),
            "n_videos_with_mentions": n_videos_with_mentions,
            "confound_tier": confound_tier(share),
        }
    return scores


def build_summary(counts, pct, confound_scores):
    rows = []
    for artist in counts.index:
        topic_counts = {t: counts.loc[artist, t] if t in counts.columns else 0 for t in TOPICS}
        dominant = max(topic_counts, key=topic_counts.get)
        n = topic_counts[dominant]
        if n == 0:
            continue
        cscore = confound_scores.get((artist, dominant), {})
        rows.append({
            "artist": artist,
            "dominant_topic": dominant,
            "count": n,
            "pct_of_total": round(pct.loc[artist, dominant], 2) if dominant in pct.columns else None,
            "confidence": confidence_tier(n),
            "max_video_share": cscore.get("max_video_share", None),
            "n_videos_with_mentions": cscore.get("n_videos_with_mentions", None),
            "confound_check": cscore.get("confound_tier", "（動画別データ不足）"),
            "previously_validated": PREVIOUSLY_VALIDATED.get((artist, dominant), "（今回が初検出・他データセットでは未確認）"),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def visualize(summary_df, pct):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                          "axes.edgecolor": "#CCCCCC", "axes.grid": True, "grid.alpha": 0.3, "font.size": 11})

    CONF_COLORS = {"◎ 信頼できる": "#2EC4B6", "▲ 要注意（少数）": "#FF9F1C", "✗ 不十分": "#CCCCCC"}
    CONFOUND_EDGE = {"✅ 複数動画に分散": "#2A9D8F", "△ 中間": "#E9C46A", "⚠ 単一動画に集中": "#E63946", None: "#999999"}

    df = summary_df.copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = [CONF_COLORS.get(c, "#999") for c in df["confidence"]]
    edge_colors = [CONFOUND_EDGE.get(c, "#999") for c in df["confound_check"]]
    bars = ax.bar(range(len(df)), df["pct_of_total"], color=colors, edgecolor=edge_colors, linewidth=3)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["pct_of_total"] + 0.5, f"n={row['count']}", ha="center", fontsize=8.5)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{a.replace('$', chr(92)+'$')}\n{t}" for a, t in zip(df["artist"], df["dominant_topic"])],
                        rotation=0, ha="center", fontsize=8.5)
    ax.set_ylabel("全コメント中の比率(%)")
    ax.set_title("アーティスト別 支配的トピックの最終サマリー\n"
                 "塗り色=信頼度（緑◎/オレンジ▲/グレー✗）　枠色=交絡チェック（緑=分散/赤=単一動画集中）",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legend1 = [Patch(facecolor=c, label=l) for l, c in CONF_COLORS.items()]
    legend2 = [Patch(facecolor="white", edgecolor=c, linewidth=3, label=l)
               for l, c in CONFOUND_EDGE.items() if l]
    ax.legend(handles=legend1 + legend2, fontsize=8, loc="upper right", ncol=1)

    plt.tight_layout()
    out_path = f"{FIG_DIR}/fig_final_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"図を保存: {out_path}")


def generate_report(summary_df):
    md = "# HIPHOPファン言説分析 ― 最終サマリー（自動生成）\n\n"
    md += "run_all_v5.py の出力に、これまでの検証ルール（信頼度判定・交絡自動検出）を適用した結果。\n\n"
    md += "## サマリー表\n\n"
    md += "| アーティスト | 支配的トピック | 件数 | 比率 | 信頼度 | 交絡チェック | 過去の検証歴 |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for _, r in summary_df.iterrows():
        md += (f"| {r['artist']} | {r['dominant_topic']} | {r['count']} | {r['pct_of_total']}% | "
                f"{r['confidence']} | {r['confound_check']} | {r['previously_validated']} |\n")

    md += "\n## 自動判定ロジックについて\n\n"
    md += ("- **信頼度**: 支配的トピックの件数が50件以上なら◎、10〜49件なら▲、10件未満なら✗\n"
           "- **交絡チェック**: そのトピックの言及のうち、最大1本の動画が占める割合を計算。\n"
           "  70%以上が1本に集中していれば⚠（交絡の疑い）、50%未満なら✅（複数動画に分散・信頼できる）\n\n")

    md += "## 総合判定：今回の結果で『確実』と言える発見\n\n"
    strong = summary_df[(summary_df["confidence"] == "◎ 信頼できる") &
                         (summary_df["confound_check"] == "✅ 複数動画に分散")]
    if not strong.empty:
        for _, r in strong.iterrows():
            md += f"- **{r['artist']}（{r['dominant_topic']}）**: {r['count']}件、複数動画に分散。最も確実。\n"
    else:
        md += "- 今回の自動判定基準で「件数十分」かつ「複数動画に分散」の両方を満たすものはなかった。各行の詳細を個別に確認すること。\n"

    md += "\n## 要注意：件数は十分だが交絡の疑いがあるもの\n\n"
    caution = summary_df[(summary_df["confidence"] == "◎ 信頼できる") &
                          (summary_df["confound_check"] == "⚠ 単一動画に集中")]
    if not caution.empty:
        for _, r in caution.iterrows():
            md += f"- **{r['artist']}（{r['dominant_topic']}）**: {r['count']}件だが、特定の1本の動画に集中している可能性。動画タイトルを個別確認すること。\n"
    else:
        md += "- 該当なし。\n"

    with open(f"{RESULT_DIR}/FINAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nレポートを保存: {RESULT_DIR}/FINAL_REPORT.md")
    print("\n" + md)


def main():
    if not os.path.isdir(RESULT_DIR):
        print(f"[エラー] {RESULT_DIR} が見つかりません。run_all_v5.py を先に実行してください。")
        sys.exit(1)

    counts, pct, confound_df = load_data()
    confound_scores = compute_confound_scores(confound_df)
    summary_df = build_summary(counts, pct, confound_scores)
    summary_df.to_csv(f"{RESULT_DIR}/final_summary.csv", index=False)

    try:
        visualize(summary_df, pct)
    except Exception as e:
        print(f"可視化でエラー(本筋には影響なし): {e}")

    generate_report(summary_df)
    print("\n完了。FINAL_REPORT.md と fig_final_summary.png を確認してください。")


if __name__ == "__main__":
    main()
