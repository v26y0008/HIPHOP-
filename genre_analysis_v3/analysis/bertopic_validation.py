"""
bertopic_validation.py (genre_v3, 頑健性チェック)

clustering_no_label.py (TF-IDF+LSA+HDBSCAN, ARI≈0という「最重要」結果)を、
文脈を考慮した埋め込みベースのBERTopicで再検証する。

「TF-IDFという浅い特徴量だからジャンル非依存に見えただけではないか」という
批判に対応するため、文脈考慮型モデル(all-MiniLM-L6-v2)でも同じ独立性
(ARI≈0)が再現されるかを確認する。

サンプリング: 全92,282件は重いため、genre別に最大3000件をサンプリング
(4ジャンル×最大3000件=最大12,000件)。

使い方:
  python bertopic_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_patterns import build_removal_pattern, strip_names

GENRE_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = GENRE_ROOT / "data" / "processed" / "cluster_assignments_genre_v3.csv"
OUT_DIR = GENRE_ROOT / "data" / "processed"

RANDOM_STATE = 42
MAX_PER_GENRE = 3000


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先にclustering_no_label.pyを実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    print(f"元データ: {len(df):,}件")

    # genre別に最大MAX_PER_GENRE件をサンプリング
    parts = []
    for genre, group in df.groupby("genre"):
        parts.append(group.sample(n=min(MAX_PER_GENRE, len(group)), random_state=RANDOM_STATE))
    sample_df = pd.concat(parts, ignore_index=True)
    print(f"サンプリング後: {len(sample_df):,}件")
    print(sample_df["genre"].value_counts())

    # clustering_no_label.py と同じ固有名詞除去を適用(公平な比較のため)
    pattern = build_removal_pattern()
    sample_df["text_cluster"] = sample_df["text"].apply(lambda t: strip_names(t, pattern))
    sample_df = sample_df[sample_df["text_cluster"].str.split().str.len() >= 4].reset_index(drop=True)
    print(f"固有名詞除去・短文除去後: {len(sample_df):,}件")

    docs = sample_df["text_cluster"].tolist()

    print("\nSentenceTransformer(all-MiniLM-L6-v2)で埋め込み計算中...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)

    print("\nBERTopicでクラスタリング中...")
    topic_model = BERTopic(embedding_model=embedding_model, language="english",
                            calculate_probabilities=False, verbose=True, min_topic_size=50)
    topics, _ = topic_model.fit_transform(docs, embeddings)
    sample_df["bertopic_topic"] = topics

    topic_info = topic_model.get_topic_info()
    print("\n=== BERTopicトピック一覧(上位20) ===")
    print(topic_info.head(20).to_string())

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"\nトピック数: {n_topics} (outlier: {n_outliers:,}/{len(topics):,})")

    # --- ARI比較 ---
    valid = sample_df[sample_df["bertopic_topic"] != -1]
    ari_genre = adjusted_rand_score(valid["genre"], valid["bertopic_topic"])
    ari_artist = adjusted_rand_score(valid["artist_name"], valid["bertopic_topic"])
    ari_hdbscan = adjusted_rand_score(valid["cluster"], valid["bertopic_topic"])

    print(f"\nAdjusted Rand Index (BERTopic vs genre)          : {ari_genre:.4f}")
    print(f"Adjusted Rand Index (BERTopic vs artist_name)     : {ari_artist:.4f}")
    print(f"Adjusted Rand Index (BERTopic vs 既存HDBSCAN)     : {ari_hdbscan:.4f}")
    print("(前回のTF-IDF+HDBSCAN結果: ARI(cluster vs genre)=-0.0016, (cluster vs artist)=0.0004)")

    ct = pd.crosstab(valid["bertopic_topic"], valid["genre"])
    print("\nBERTopicトピック x genre クロス集計(上位10トピック):")
    print(ct.head(10))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.drop(columns=["text_cluster"]).to_csv(OUT_DIR / "bertopic_assignments.csv", index=False)
    topic_info.to_csv(OUT_DIR / "bertopic_topic_info.csv", index=False)

    summary = pd.DataFrame([{
        "n_sample": len(sample_df), "n_topics": n_topics, "n_outliers": n_outliers,
        "ari_bertopic_vs_genre": ari_genre, "ari_bertopic_vs_artist": ari_artist,
        "ari_bertopic_vs_hdbscan": ari_hdbscan,
    }])
    summary.to_csv(OUT_DIR / "bertopic_validation_summary.csv", index=False)
    print(f"\n保存完了: {OUT_DIR}")


if __name__ == "__main__":
    main()
