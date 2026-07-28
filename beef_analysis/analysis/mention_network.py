"""
mention_network.py (v2 追加モジュール)

ビーフ曲コメントで「誰が誰の名前を言及しているか」をネットワークとして可視化する。

ノード：アーティスト（言及された側）
エッジ：同一コメント内での共起言及（重み=共起回数）

分析：
  1. 中心性（誰がコメント空間で最も言及されるか）
  2. カタログ曲 vs ビーフ曲でネットワーク構造が変わるか
  3. ビーフ別ネットワーク（当事者が中心になるか）
  4. Louvain法によるコミュニティ検出

注意（beef_idの対応関係）: このプロジェクトの内部beef_idは
  b1=Kendrick vs Drake, b2=Pusha T vs Drake, b4=Eminem vs MGK,
  b5=Lil Wayne vs Birdman, b6=Megan vs Nicki
であり、b3=Drake vs Meek Millはコメント欄無効のため分析から除外済み
（欠番のまま、b1→b2→b4という順序でb3は存在しない）。
外部ドキュメントがb1-b5のように連番で呼んでいる場合は要注意
（過去にもポスターPPTXで同種の取り違えが発生している）。

使い方:
  python mention_network.py
"""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community as nx_community
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Meiryo"

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

# アーティスト名の別称 → 正規化名（このプロジェクトのartist_name表記に合わせる）
ARTIST_ALIASES = {
    "kendrick": "Kendrick Lamar", "kdot": "Kendrick Lamar", "k dot": "Kendrick Lamar",
    "drake": "Drake", "aubrey": "Drake", "drizzy": "Drake", "champagne papi": "Drake",
    "pusha": "Pusha T", "pusha t": "Pusha T", "king push": "Pusha T",
    "eminem": "Eminem", "slim shady": "Eminem", "marshall": "Eminem",
    "mgk": "MGK", "machine gun kelly": "MGK",
    "lil wayne": "Lil Wayne", "weezy": "Lil Wayne", "tunechi": "Lil Wayne",
    "birdman": "Birdman", "baby (rapper)": "Birdman",
    "megan": "Megan Thee Stallion", "thee stallion": "Megan Thee Stallion",
    "nicki": "Nicki Minaj", "minaj": "Nicki Minaj", "barbie": "Nicki Minaj", "onika": "Nicki Minaj",
    # 関連言及されやすいその他の主要アーティスト
    "jay z": "Jay-Z", "jay-z": "Jay-Z", "hov": "Jay-Z", "jigga": "Jay-Z",
    "nas": "Nas", "nasir": "Nas",
    "j cole": "J. Cole",
    "future": "Future",
    "travis scott": "Travis Scott", "travis": "Travis Scott",
    "meek mill": "Meek Mill", "meek": "Meek Mill",
    "rick ross": "Rick Ross",
    "kanye": "Kanye West", "ye": "Kanye West",
}

BEEF_PARTIES = {
    "b1": ["Kendrick Lamar", "Drake"],
    "b2": ["Pusha T", "Drake"],
    "b4": ["Eminem", "MGK"],
    "b5": ["Lil Wayne", "Birdman"],
    "b6": ["Megan Thee Stallion", "Nicki Minaj"],
    # b7 (Ice Cube vs NWA) はbeefカテゴリのコメントが両アーティストとも
    # コメント欄無効のためbuild_dataset.pyの時点で除外済み（詳細はconfig/songs_v2.csv）
}


def extract_mentions(text):
    text_lower = f" {str(text).lower()} "
    mentioned = set()
    for alias, canonical in ARTIST_ALIASES.items():
        if f" {alias} " in text_lower or text_lower.startswith(f" {alias} ") or text_lower.endswith(f" {alias} "):
            mentioned.add(canonical)
    return list(mentioned)


def build_network(df, track_category=None, beef_id=None):
    if track_category:
        df = df[df["track_category"] == track_category]
    if beef_id:
        df = df[df["beef_id"] == beef_id]

    G = nx.Graph()
    co_occurrence = defaultdict(int)
    single_mention = defaultdict(int)

    for text in df["text_clean"]:
        mentions = list(set(extract_mentions(text)))
        for artist in mentions:
            single_mention[artist] += 1
            if artist not in G:
                G.add_node(artist)
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                a, b = sorted([mentions[i], mentions[j]])
                co_occurrence[(a, b)] += 1

    for (a, b), weight in co_occurrence.items():
        if weight >= 3:
            G.add_edge(a, b, weight=weight)

    nx.set_node_attributes(G, single_mention, "mention_count")
    return G


def analyze_centrality(G):
    if len(G.nodes) == 0:
        return pd.DataFrame()
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    mention = nx.get_node_attributes(G, "mention_count")

    df = pd.DataFrame({
        "artist": list(degree.keys()),
        "degree": list(degree.values()),
        "betweenness": [betweenness.get(n, 0) for n in degree.keys()],
        "mention_count": [mention.get(n, 0) for n in degree.keys()],
    }).sort_values("mention_count", ascending=False)
    return df


def detect_communities(G):
    if community_louvain is None or len(G.nodes) == 0 or len(G.edges) == 0:
        return {}
    return community_louvain.best_partition(G, weight="weight")


def compute_network_metrics(G, label=""):
    """density / avg_degree / modularity / n_communities を算出（NetworkX組み込みのlouvain_communitiesを使用）"""
    if len(G.nodes) == 0:
        print(f"--- {label} --- ノードなし、スキップ")
        return {"label": label, "n_nodes": 0, "n_edges": 0, "density": None,
                "avg_degree": None, "max_degree_node": None, "modularity": None, "n_communities": None}

    density = nx.density(G)
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees) if len(degrees) > 0 else 0
    max_degree_node = max(degrees, key=degrees.get) if degrees else None

    if len(G.edges) > 0:
        communities = nx_community.louvain_communities(G, weight="weight", seed=42)
        modularity = nx_community.modularity(G, communities, weight="weight")
        n_communities = len(communities)
    else:
        modularity, n_communities = None, len(G.nodes)

    print(f"--- {label} ---")
    print(f"Nodes: {len(G.nodes)}  Edges: {len(G.edges)}")
    print(f"Density: {density:.4f}")
    print(f"Avg Degree: {avg_degree:.2f}")
    print(f"Max Degree Node: {max_degree_node} ({degrees.get(max_degree_node)})")
    print(f"Modularity: {modularity:.4f}" if modularity is not None else "Modularity: N/A")
    print(f"Number of communities: {n_communities}")

    return {
        "label": label, "n_nodes": len(G.nodes), "n_edges": len(G.edges),
        "density": density, "avg_degree": avg_degree,
        "max_degree_node": max_degree_node,
        "modularity": modularity, "n_communities": n_communities,
    }


def plot_network(G, partition, title, output_path, beef_parties=None):
    if len(G.nodes) == 0:
        print(f"[スキップ] ノードなし: {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2)

    mention_counts = nx.get_node_attributes(G, "mention_count")
    node_sizes = [max(mention_counts.get(n, 1) * 3, 100) for n in G.nodes]

    if partition:
        n_communities = max(partition.values()) + 1
        cmap = plt.cm.Set1(np.linspace(0, 1, max(n_communities, 2)))
        node_colors = [cmap[partition.get(n, 0)] for n in G.nodes]
    elif beef_parties:
        node_colors = ["#CC4400" if n in beef_parties else "#888888" for n in G.nodes]
    else:
        node_colors = ["#1E6FBE"] * len(G.nodes)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_w * 5 for w in edge_weights]

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edge_color="#CCCCCC", width=edge_widths,
        font_size=9, font_weight="bold",
        with_labels=True,
    )
    ax.set_title(title, fontsize=13, pad=15)
    ax.axis("off")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure保存: {output_path}")


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])

    print("=== ビーフ曲 vs カタログ曲 ネットワーク比較 ===")
    G_beef = build_network(df, track_category="beef")
    G_catalog = build_network(df, track_category="catalog")

    cent_beef = analyze_centrality(G_beef)
    cent_catalog = analyze_centrality(G_catalog)

    print(f"\n【ビーフ曲】ノード数={G_beef.number_of_nodes()}, エッジ数={G_beef.number_of_edges()}")
    print(cent_beef.head(10).to_string())
    print(f"\n【カタログ曲】ノード数={G_catalog.number_of_nodes()}, エッジ数={G_catalog.number_of_edges()}")
    print(cent_catalog.head(10).to_string())

    partition_beef = detect_communities(G_beef)
    if community_louvain is None:
        print("[警告] python-louvain が見つからないためコミュニティ検出をスキップ")

    print("\n=== ネットワーク指標比較（density / modularity / avg degree）===")
    metrics_results = [
        compute_network_metrics(G_beef, "Beef tracks"),
        compute_network_metrics(G_catalog, "Catalog tracks"),
    ]

    plot_network(G_beef, partition_beef,
                 "ビーフ曲コメントにおけるアーティスト言及ネットワーク",
                 FIGURES_DIR / "fig_network_beef.png")
    plot_network(G_catalog, {},
                 "カタログ曲コメントにおけるアーティスト言及ネットワーク",
                 FIGURES_DIR / "fig_network_catalog.png")

    print("\n=== ビーフ別ネットワーク ===")
    beef_summaries = []
    for bid, parties in BEEF_PARTIES.items():
        G_b = build_network(df, track_category="beef", beef_id=bid)
        cent = analyze_centrality(G_b)
        print(f"\n{bid} ({' vs '.join(parties)}): ノード数={G_b.number_of_nodes()}, エッジ数={G_b.number_of_edges()}")
        print(cent.head(5).to_string())
        plot_network(G_b, {}, f"{bid}: {' vs '.join(parties)} ビーフ曲ネットワーク",
                     FIGURES_DIR / f"fig_network_{bid}.png",
                     beef_parties=set(parties))
        if not cent.empty:
            cent["beef_id"] = bid
            beef_summaries.append(cent)

    print("\n=== ビーフ当事者の中心性（全ビーフ曲ネットワーク統合）===")
    all_parties = set(sum(BEEF_PARTIES.values(), []))
    if not cent_beef.empty:
        cent_beef["is_party"] = cent_beef["artist"].isin(all_parties)
        party_cent = cent_beef[cent_beef["is_party"]]["mention_count"].mean()
        nonparty = cent_beef[~cent_beef["is_party"]]
        nonparty_cent = nonparty["mention_count"].mean() if len(nonparty) else float("nan")
        print(f"当事者の平均言及数: {party_cent:.1f}")
        print(f"非当事者の平均言及数: {nonparty_cent:.1f}" if not np.isnan(nonparty_cent) else "非当事者の平均言及数: 該当ノードなし")
        if not np.isnan(nonparty_cent) and nonparty_cent > 0:
            print(f"比率: {party_cent / nonparty_cent:.1f}倍")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cent_beef.to_csv(OUT_DIR / "network_centrality_beef.csv", index=False)
    cent_catalog.to_csv(OUT_DIR / "network_centrality_catalog.csv", index=False)
    if beef_summaries:
        pd.concat(beef_summaries, ignore_index=True).to_csv(OUT_DIR / "network_centrality_by_beef.csv", index=False)

    pd.DataFrame(metrics_results).to_csv(OUT_DIR / "network_metrics_comparison.csv", index=False)
    print(f"\n保存: {OUT_DIR / 'network_metrics_comparison.csv'}")


if __name__ == "__main__":
    main()
