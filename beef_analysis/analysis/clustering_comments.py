"""
clustering_comments.py (v2 追加モジュール)

コメントをTF-IDF + TruncatedSVD(LSA) + KMeansでクラスタリングし、
カテゴリ（catalog / beef / post）ごとのクラスタ分布を比較する。
カイ二乗検定（語彙の有無）とは独立に、「ビーフがコメントのトピック構造
そのものを変えるか」をデータ駆動で検証する。

手順:
  1. TF-IDFでベクトル化
  2. TruncatedSVD(LSA)で次元圧縮
  3. KMeansでクラスタリング
  4. 各クラスタの上位語彙を確認
  5. カテゴリ別クラスタ分布をカイ二乗検定で比較
  6. クラスタ数 k=3〜7 で感度分析
  7. UMAPで2次元可視化

使い方:
  python clustering_comments.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

matplotlib.rcParams["font.family"] = "sans-serif"
warnings.filterwarnings("ignore")

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

N_COMPONENTS = 50
N_CLUSTERS = 5
N_TOP_WORDS = 15
RANDOM_STATE = 42

CATEGORY_ORDER = ["catalog", "beef", "post"]
CAT_COLORS = {"catalog": "#999999", "beef": "#0072B2", "post": "#E69F00"}


def load_data():
    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df = df[df["track_category"].isin(CATEGORY_ORDER)].reset_index(drop=True)
    print(f"総コメント数: {len(df):,}")
    print(df.groupby(["beef_id", "track_category"]).size().unstack(fill_value=0))
    return df


def vectorize(df):
    vectorizer = TfidfVectorizer(
        max_features=8000, min_df=5, max_df=0.95,
        ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["text_clean"].fillna(""))
    print(f"TF-IDF行列: {X.shape}")
    return X, vectorizer


def reduce_dim(X, n_components=N_COMPONENTS):
    n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_lsa = svd.fit_transform(X)
    X_lsa = normalize(X_lsa)
    print(f"LSA後: {X_lsa.shape}, 累積説明分散: {svd.explained_variance_ratio_.sum():.3f}")
    return X_lsa, svd


def cluster(X_lsa, n_clusters=N_CLUSTERS):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_lsa)
    print(f"クラスタ分布: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, km


def get_top_words(km, svd, vectorizer, n_top=N_TOP_WORDS):
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    top_words = {}
    print("\n=== クラスタ別上位語彙 ===")
    for i in range(km.n_clusters):
        words = [terms[ind] for ind in order_centroids[i, :n_top]]
        top_words[i] = words
        print(f"  クラスタ {i}: {', '.join(words[:10])}")
    return top_words


def analyze_cluster_distribution(df, labels):
    df = df.copy()
    df["cluster"] = labels

    ct = pd.crosstab(df["track_category"], df["cluster"]).reindex(CATEGORY_ORDER)
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100

    print("\n=== カテゴリ別クラスタ分布（%） ===")
    print(ct_norm.round(1))

    chi2, p, dof, expected = chi2_contingency(ct)
    v = association(ct, method="cramer")
    print(f"\nカイ二乗検定: chi2={chi2:.2f}, p={p:.4e}, V={v:.3f}")

    print("\n=== ビーフ曲で比率が高いクラスタ（カタログ曲比） ===")
    for c in ct_norm.columns:
        diff = ct_norm.loc["beef", c] - ct_norm.loc["catalog", c]
        if diff > 5:
            print(f"  クラスタ {c}: beef={ct_norm.loc['beef', c]:.1f}%, "
                  f"catalog={ct_norm.loc['catalog', c]:.1f}% (+{diff:.1f}pp)")

    ct_norm.to_csv(OUT_DIR / "cluster_distribution.csv")
    return ct, ct_norm, chi2, p, v


def sensitivity_analysis(df, X_lsa):
    print("\n=== 感度分析：クラスタ数 k=3〜7 ===")
    results = []
    for k in range(3, 8):
        km_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_k = km_k.fit_predict(X_lsa)
        df_k = df.copy()
        df_k["cluster"] = labels_k
        ct_k = pd.crosstab(df_k["track_category"], df_k["cluster"]).reindex(CATEGORY_ORDER)
        chi2_k, p_k, _, _ = chi2_contingency(ct_k)
        v_k = association(ct_k, method="cramer")

        ct_norm_k = ct_k.div(ct_k.sum(axis=1), axis=0) * 100
        max_diff = (ct_norm_k.loc["beef"] - ct_norm_k.loc["catalog"]).max()

        results.append({"k": k, "chi2": chi2_k, "p": p_k, "V": v_k, "max_diff_pp": max_diff})
        print(f"  k={k}: chi2={chi2_k:.1f}, p={p_k:.2e}, V={v_k:.3f}, max_diff={max_diff:.1f}pp")

    return pd.DataFrame(results)


def plot_cluster_distribution(ct_norm, top_words):
    cluster_names = {i: f"C{i}\n({', '.join(top_words[i][:3])})" for i in top_words}

    fig, ax = plt.subplots(figsize=(9, 5))
    cat_labels = ["Catalog", "Beef", "Post"]
    colors = plt.cm.Set2(np.linspace(0, 1, ct_norm.shape[1]))

    bottom = np.zeros(3)
    for ci, col in enumerate(ct_norm.columns):
        vals = [ct_norm.loc[c, col] if c in ct_norm.index else 0 for c in CATEGORY_ORDER]
        ax.bar(cat_labels, vals, bottom=bottom, color=colors[ci],
               label=cluster_names.get(col, f"C{col}"), width=0.5)
        bottom += np.array(vals)

    ax.set_ylabel("Share of comments (%)")
    ax.set_title("Comment topic-cluster distribution by track category", loc="left")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig9_cluster_dist_v2.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure 保存: {out_path}")


def plot_umap(X_lsa, df, labels):
    try:
        import umap
    except ImportError:
        print("umap-learn がインストールされていません。スキップします。")
        return

    print("UMAP計算中...")
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15)
    n_sample = min(5000, len(X_lsa))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_lsa), n_sample, replace=False)
    X_2d = reducer.fit_transform(X_lsa[idx])

    df_sub = df.iloc[idx].copy()
    df_sub["cluster"] = labels[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for cat, color in CAT_COLORS.items():
        mask = df_sub["track_category"] == cat
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=3, alpha=0.4, label=cat)
    axes[0].set_title("By track category")
    axes[0].legend(markerscale=3)

    cluster_colors = plt.cm.Set1(np.linspace(0, 1, N_CLUSTERS))
    for ci in range(N_CLUSTERS):
        mask = df_sub["cluster"] == ci
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cluster_colors[ci]],
                         s=3, alpha=0.4, label=f"C{ci}")
    axes[1].set_title("By cluster")
    axes[1].legend(markerscale=3)

    fig.suptitle("UMAP projection (TF-IDF + LSA)", fontsize=13)
    fig.tight_layout()
    out_path = FIGURES_DIR / "fig10_cluster_umap_v2.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP Figure 保存: {out_path}")


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に preprocess.py を実行してください。")
        return

    df = load_data()
    X, vectorizer = vectorize(df)
    X_lsa, svd = reduce_dim(X)
    labels, km = cluster(X_lsa)
    top_words = get_top_words(km, svd, vectorizer)

    ct, ct_norm, chi2, p, v = analyze_cluster_distribution(df, labels)
    sensitivity_df = sensitivity_analysis(df, X_lsa)
    sensitivity_df.to_csv(OUT_DIR / "cluster_sensitivity.csv", index=False)

    plot_cluster_distribution(ct_norm, top_words)
    plot_umap(X_lsa, df, labels)

    top_words_df = pd.DataFrame({f"cluster_{k}": v for k, v in top_words.items()})
    top_words_df.to_csv(OUT_DIR / "cluster_top_words.csv", index=False)
    print(f"\n上位語彙 保存: {OUT_DIR / 'cluster_top_words.csv'}")

    print("\n" + "=" * 60)
    print("=== クラスタリング分析サマリー ===")
    print(f"クラスタ数: {N_CLUSTERS}")
    print(f"カテゴリ×クラスタ のカイ二乗検定: p={p:.4e}, V={v:.3f}")


if __name__ == "__main__":
    main()
