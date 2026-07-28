"""
beef_dendrogram.py (v2 追加モジュール)

5件のビーフ（b1, b2, b4, b5, b6 — b3はコメント欄無効のため除外済み）を
特徴量ベクトルで階層型クラスタリング（Ward法）する。
n=5と小さいためKMeans等ではなくデンドログラムによる記述的な構造把握に用いる。

特徴量: 対立語彙率・相手名言及率・返信数平均・いいね数平均・平均コメント長・再生効率
対象  : 各ビーフのbeef曲カテゴリのコメント/動画

問い：ビーフには複数のタイプ（ラップバトル型 vs 契約破局型）が
      データ駆動でも分離するか？

使い方:
  python beef_dendrogram.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Meiryo"

BEEF_ROOT = Path(__file__).resolve().parents[1]
WIN_LOSS_COMMENTS = BEEF_ROOT / "data" / "processed" / "comments_with_win_loss_v2.csv"
EFFICIENCY = BEEF_ROOT / "data" / "processed" / "video_stats_with_efficiency_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

BEEF_LABELS = {
    "b1": "b1: Kendrick vs Drake",
    "b2": "b2: Pusha T vs Drake",
    "b4": "b4: Eminem vs MGK",
    "b5": "b5: Wayne vs Birdman",
    "b6": "b6: Megan vs Nicki",
    # b7 (Ice Cube vs NWA) は両アーティストのbeefカテゴリがコメント欄無効のため除外済み
}

FEATURES = ["conflict_rate", "opponent_rate", "reply_rate", "like_rate", "avg_length", "view_efficiency"]


def main():
    df = pd.read_csv(WIN_LOSS_COMMENTS, keep_default_na=False, na_values=[""])
    beef_only = df[df["track_category"] == "beef"]

    beef_features = beef_only.groupby("beef_id").agg(
        conflict_rate=("has_conflict", "mean"),
        opponent_rate=("has_opponent", "mean"),
        reply_rate=("reply_count", "mean"),
        like_rate=("like_count", "mean"),
        avg_length=("text_clean", lambda x: x.str.len().mean()),
    ).reset_index()

    eff = pd.read_csv(EFFICIENCY, keep_default_na=False, na_values=[""])
    eff_beef = eff[eff["track_category"] == "beef"].groupby("beef_id").agg(
        view_efficiency=("view_efficiency", "mean"),
    ).reset_index()
    beef_features = beef_features.merge(eff_beef, on="beef_id", how="left")

    print(f"ビーフ数: {len(beef_features)}")
    print(beef_features.to_string())

    X = StandardScaler().fit_transform(beef_features[FEATURES].fillna(0))

    Z = linkage(X, method="ward")

    labels = [BEEF_LABELS.get(b, b) for b in beef_features["beef_id"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    dendrogram(Z, labels=labels, ax=ax, color_threshold=2.0, leaf_font_size=11)
    ax.set_title(f"ビーフの階層型クラスタリング（Ward法、n={len(beef_features)}）")
    ax.set_ylabel("距離")
    ax.set_xlabel("ビーフ")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_beef_dendrogram.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure保存: {out_path}")

    for k in [2, 3]:
        beef_features[f"cluster_k{k}"] = fcluster(Z, k, criterion="maxclust")

    beef_features["beef_type"] = beef_features["beef_id"].map({
        "b1": "rap_battle", "b2": "rap_battle", "b4": "rap_battle",
        "b5": "contract_breakdown", "b6": "rap_battle",
    })

    print("\nn=5と小標本のため、デンドログラムの解釈は記述的な傾向把握にとどめる:")
    print(beef_features[["beef_id", "beef_type", "cluster_k2", "cluster_k3"]].to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beef_features.to_csv(OUT_DIR / "beef_clusters.csv", index=False)
    print(f"\n保存: {OUT_DIR / 'beef_clusters.csv'}")


if __name__ == "__main__":
    main()
