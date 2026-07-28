"""
clustering_videos.py (v2 追加モジュール、補足分析)

動画レベルの指標（対立語彙率・返信率・いいね率・平均コメント長・熱量）で
KMeansクラスタリングを行い、カタログ/ビーフ/POST曲が自然に分離されるかを確認する。

n が小さい（コメント分析対象の動画のみ、約26本）ため、あくまで補足的な分析として
位置づける。comment_per_view はengagement_by_video.csv（旧v1系だが構造は同じ）ではなく、
v2の video_level_analysis_v2.csv（engagement_analysis.pyの出力）を使う。

使い方:
  python clustering_videos.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "sans-serif"

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "video_level_analysis_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

RANDOM_STATE = 42
N_CLUSTERS = 3
CAT_COLORS = {"catalog": "#999999", "beef": "#0072B2", "post": "#E69F00"}
CLUSTER_COLORS = ["#1A9E6A", "#CC4400", "#1E6FBE"]

FEATURES = ["pct_conflict", "reply_rate", "like_rate", "avg_length", "comment_per_view"]


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に engagement_analysis.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH)
    # engagement_analysis.py の出力には pct_conflict が無いため、comments_clean_v2.csv から補完
    if "pct_conflict" not in df.columns:
        comments_path = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
        comments = pd.read_csv(comments_path, keep_default_na=False, na_values=[""])
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from conflict_vocab_analysis import has_conflict_vocab
        comments["has_conflict"] = comments["text_clean"].fillna("").apply(has_conflict_vocab)
        pct_by_video = comments.groupby("video_id")["has_conflict"].mean().mul(100).rename("pct_conflict")
        df = df.merge(pct_by_video, on="video_id", how="left")

    print(f"動画数: {len(df)}")
    print(df.groupby("track_category").size())

    df_clean = df.dropna(subset=FEATURES).copy()
    print(f"有効動画数: {len(df_clean)}")

    if len(df_clean) < N_CLUSTERS * 2:
        print("有効動画数が少なすぎるためクラスタリングをスキップします。")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[FEATURES])

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df_clean["cluster"] = km.fit_predict(X)

    ct = pd.crosstab(df_clean["cluster"], df_clean["track_category"])
    print("\nクラスタ × カテゴリ:")
    print(ct)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    df_clean["pc1"] = X_2d[:, 0]
    df_clean["pc2"] = X_2d[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for cat, color in CAT_COLORS.items():
        mask = df_clean["track_category"] == cat
        axes[0].scatter(df_clean.loc[mask, "pc1"], df_clean.loc[mask, "pc2"],
                         c=color, s=80, label=cat, edgecolors="white")
    axes[0].set_title("By track category")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].legend()

    for ci in range(N_CLUSTERS):
        mask = df_clean["cluster"] == ci
        axes[1].scatter(df_clean.loc[mask, "pc1"], df_clean.loc[mask, "pc2"],
                         c=CLUSTER_COLORS[ci], s=80, label=f"Cluster {ci}", edgecolors="white")
    axes[1].set_title(f"KMeans cluster (k={N_CLUSTERS})")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].legend()

    fig.suptitle("Video-level clustering (PCA + KMeans) — supplementary, n={}".format(len(df_clean)), fontsize=12)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig11_video_clustering_v2.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 保存: {out_path}")

    print("\nクラスタ重心（標準化前のスケール）:")
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
    print(centers.round(4))
    centers.to_csv(OUT_DIR / "video_cluster_centers.csv", index=False)


if __name__ == "__main__":
    main()
