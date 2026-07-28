"""
clustering_no_label.py (genre_v3, 最重要分析)

「教授指摘への直接回答」: ジャンルラベルを一切使わずに（TF-IDFの入力からアーティスト名・
曲名を除去した上で）コメントテキストをクラスタリングし、得られたクラスタが
事後的に genre / artist_name と対応するかどうかを検証する。

対応する場合 → トピック構造がジャンルと結びついている
対応しない場合 → トピック構造はジャンルとは独立（ジャンルは単なるラベルで、
                  実際の話題の乗り方はジャンル横断的）

パイプライン:
  1. comments_clean_genre_v3.csv (19アーティスト) + beef_context_clean.csv (8アーティスト)
     を合体し、raw textに対して固有名詞除去(name_patterns.py)を再適用
  2. TF-IDF (max_features=5000, min_df=5, stop_words='english')
  3. TruncatedSVD で50次元に圧縮
  4. HDBSCAN でクラスタリング（ジャンルラベル不使用）
  5. UMAP で2次元に落として可視化用座標を作る（クラスタリングそのものには使わない）
  6. クラスタ x genre のクロス集計・カイ二乗検定・Adjusted Rand Index で対応関係を検証

使い方:
  python clustering_no_label.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

try:
    import hdbscan
except ImportError:
    print("先に次を実行してください: pip install hdbscan")
    sys.exit(1)

try:
    import umap
except ImportError:
    umap = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_patterns import build_removal_pattern, strip_names

matplotlib.rcParams["font.family"] = "sans-serif"

GENRE_ROOT = Path(__file__).resolve().parents[1]
GENRE_CLEAN = GENRE_ROOT / "data" / "processed" / "comments_clean_genre_v3.csv"
BEEF_CONTEXT_CLEAN = GENRE_ROOT / "data" / "processed" / "beef_context_clean.csv"
OUT_DIR = GENRE_ROOT / "data" / "processed"
FIGURES_DIR = GENRE_ROOT / "outputs" / "figures"

RANDOM_STATE = 42
MIN_CLUSTER_SIZE = 400
GENRE_COLORS = {
    "boom_bap": "#0072B2", "trap": "#E69F00", "alternative": "#009E73", "beef_context": "#999999",
}


def load_corpus():
    frames = []
    if GENRE_CLEAN.exists():
        df1 = pd.read_csv(GENRE_CLEAN, keep_default_na=False, na_values=[""])
        df1["source"] = "genre_v3"
        frames.append(df1[["comment_id", "genre", "artist_name", "track_category", "text", "source"]])
    if BEEF_CONTEXT_CLEAN.exists():
        df2 = pd.read_csv(BEEF_CONTEXT_CLEAN, keep_default_na=False, na_values=[""])
        df2["source"] = "beef_context"
        frames.append(df2[["comment_id", "genre", "artist_name", "track_category", "text", "source"]])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_corpus()
    if df is None:
        print("エラー: comments_clean_genre_v3.csv / beef_context_clean.csv が見つかりません。")
        return

    print(f"合体前コメント数: {len(df):,}")
    print(df.groupby(["genre", "source"]).size())

    pattern = build_removal_pattern()
    df["text_cluster"] = df["text"].apply(lambda t: strip_names(t, pattern))
    df = df[df["text_cluster"].str.split().str.len() >= 4].reset_index(drop=True)
    print(f"\n固有名詞除去・短文除去後: {len(df):,}件")

    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, stop_words="english")
    X_tfidf = vectorizer.fit_transform(df["text_cluster"])
    print(f"TF-IDF行列: {X_tfidf.shape}")

    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_tfidf)
    print(f"LSA(50次元)説明分散比の合計: {svd.explained_variance_ratio_.sum():.3f}")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
    labels = clusterer.fit_predict(X_svd)
    df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\nHDBSCANクラスタ数: {n_clusters} (ノイズ点: {n_noise:,} / {len(df):,})")
    print(pd.Series(labels).value_counts().sort_index())

    # --- クラスタ x genre 対応関係の検証 ---
    valid = df[df["cluster"] != -1]
    ct = pd.crosstab(valid["cluster"], valid["genre"])
    print("\nクラスタ x genre クロス集計:")
    print(ct)

    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
        print(f"\nカイ二乗検定: chi2={chi2:.2f}, p={p:.4g}, Cramer's V={cramers_v:.3f}")
    else:
        chi2, p, cramers_v = None, None, None
        print("\nクロス集計の次元が不足しておりカイ二乗検定をスキップ")

    ari_genre = adjusted_rand_score(valid["genre"], valid["cluster"])
    ari_artist = adjusted_rand_score(valid["artist_name"], valid["cluster"])
    print(f"\nAdjusted Rand Index (cluster vs genre)      : {ari_genre:.4f}")
    print(f"Adjusted Rand Index (cluster vs artist_name) : {ari_artist:.4f}")
    print("(ARIは0に近いほど「クラスタはgenre/artistとは独立」、1に近いほど「クラスタ=genre/artist」)")

    # --- クラスタごとの特徴語（TF-IDF平均が高い語）---
    terms = np.array(vectorizer.get_feature_names_out())
    top_words_rows = []
    for c in sorted(set(labels)):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        mean_tfidf = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
        nonzero_idx = np.flatnonzero(mean_tfidf)
        ranked = nonzero_idx[np.argsort(mean_tfidf[nonzero_idx])[::-1]]
        top_idx = ranked[:15]
        top_words_rows.append({
            "cluster": c, "n": int(mask.sum()),
            "top_words": ", ".join(terms[top_idx]),
        })
    top_words_df = pd.DataFrame(top_words_rows)
    print("\nクラスタ別特徴語:")
    for _, row in top_words_df.iterrows():
        print(f"  cluster {row['cluster']} (n={row['n']}): {row['top_words']}")

    # --- UMAP 2次元可視化 ---
    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, metric="euclidean")
        X_umap = reducer.fit_transform(X_svd)
        df["umap_x"] = X_umap[:, 0]
        df["umap_y"] = X_umap[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        for genre, color in GENRE_COLORS.items():
            mask = df["genre"] == genre
            if mask.sum() == 0:
                continue
            axes[0].scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                             c=color, s=4, alpha=0.35, label=genre, edgecolors="none")
        axes[0].set_title("UMAP projection colored by genre (not used for clustering)")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")
        axes[0].legend(markerscale=4, fontsize=8)

        cluster_ids = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(cluster_ids):
            mask = df["cluster"] == c
            color = "#CCCCCC" if c == -1 else cmap(i % 10)
            label = "noise" if c == -1 else f"cluster {c}"
            axes[1].scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                             c=[color], s=4, alpha=0.35, label=label, edgecolors="none")
        axes[1].set_title(f"HDBSCAN clusters (label-free, k={n_clusters})")
        axes[1].set_xlabel("UMAP-1")
        axes[1].set_ylabel("UMAP-2")
        axes[1].legend(markerscale=4, fontsize=8, ncol=2)

        ari_text = f"ARI(cluster vs genre)={ari_genre:.3f}"
        fig.suptitle(f"Genre-label-free clustering — {ari_text}", fontsize=12)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / "fig_genre_v3_clustering_no_label.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigure保存: {out_path}")
    else:
        print("\numap-learn が見つからないため可視化をスキップ（クラスタリング自体は完了）")

    # --- 出力 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["text_cluster"]).to_csv(OUT_DIR / "cluster_assignments_genre_v3.csv", index=False)
    top_words_df.to_csv(OUT_DIR / "cluster_top_words_genre_v3.csv", index=False)
    ct.to_csv(OUT_DIR / "cluster_by_genre_crosstab_v3.csv")

    summary = pd.DataFrame([{
        "n_comments": len(df), "n_clusters": n_clusters, "n_noise": int(n_noise),
        "chi2": chi2, "p_value": p, "cramers_v": cramers_v,
        "ari_cluster_vs_genre": ari_genre, "ari_cluster_vs_artist": ari_artist,
    }])
    summary.to_csv(OUT_DIR / "clustering_no_label_summary.csv", index=False)
    print(f"\n保存完了: {OUT_DIR}")


if __name__ == "__main__":
    main()
