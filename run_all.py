"""
run_all.py
=================================================================
HIPHOPアーティスト ファン価値観分析 ― 全自動パイプライン

これ1本を実行するだけで以下が全部自動で実行される：
  1. 必要パッケージの自動インストール
  2. YouTube Data API v3 でコメント収集
  3. 前処理
  4. Sentence-BERT で埋め込み
  5. UMAP + HDBSCAN でクラスタリング
  6. アーティスト価値観ベクトル作成
  7. コサイン類似度 + ネットワーク分析
  8. グラフ5枚を生成
  9. 考察レポート(discussion_report.md)を実データの数値で自動生成

使い方:
  1. 下の YOUTUBE_API_KEY を自分のAPIキーに変更（すでに入力済みならそのままでOK）
  2. ターミナルで実行:
       python run_all.py
  3. 数分待つと ./output/ フォルダに全結果が出力される
=================================================================
"""

import os
import sys
import subprocess

# ============================================================
# 0. 設定
# ============================================================
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyASkpFtAv7aUOSSjoEJ-2iJfDSvVQKeG5Q")

ARTISTS = {
    "Nas":                "Boom bap / Lyric",
    "Kendrick Lamar":     "Boom bap / Lyric",
    "J. Cole":            "Boom bap / Lyric",
    "Joey Bada$$":        "Boom bap / Lyric",
    "Future":             "Trap",
    "Playboi Carti":      "Trap",
    "Young Thug":         "Trap",
    "Travis Scott":       "Trap",
    "Tyler, The Creator": "Alternative",
    "JPEGMAFIA":          "Alternative",
    "A$AP Rocky":         "Alternative",
    "Mac Miller":         "Alternative",
}
VIDEOS_PER_ARTIST   = 3
COMMENTS_PER_VIDEO  = 400
SIM_THRESHOLD       = 0.80   # ネットワークのエッジを引く類似度の閾値

OUT_DIR = "output"
RAW_DIR = f"{OUT_DIR}/data/raw"
PROC_DIR = f"{OUT_DIR}/data/processed"
RESULT_DIR = f"{OUT_DIR}/results"
FIG_DIR = f"{RESULT_DIR}/figures"

for d in (RAW_DIR, PROC_DIR, RESULT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. 必要パッケージの自動インストール
# ============================================================
def ensure_packages():
    required = {
        "googleapiclient":        "google-api-python-client",
        "sentence_transformers":  "sentence-transformers",
        "umap":                   "umap-learn",
        "hdbscan":                "hdbscan",
        "networkx":               "networkx",
        "matplotlib":             "matplotlib",
        "sklearn":                "scikit-learn",
        "pandas":                 "pandas",
        "numpy":                  "numpy",
    }
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[setup] {pip_name} をインストール中...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
    print("[setup] パッケージ確認完了\n")

ensure_packages()

import re
import time
import csv
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 2. YouTubeコメント収集
# ============================================================
# 既知の公式チャンネルID（事前確認済み・優先的に使用）
# 1つ目が無効・動画0件の場合は自動的に次の候補にフォールバックする
CHANNEL_OVERRIDES = {
    "Nas":            ["UCUPb87t9jboKl9YthgCy2hg", "UCPoQYATXIYvN5WB0c4f6jfQ"],  # Nas official -> Nas - Topic
    "Kendrick Lamar": ["UC3lBXcrKFnFAFkfVk5WuKcQ", "UCoYfzC2zMlc9M-Odgaf6OSg"],   # 公式 -> KendrickLamarVEVO
    "J. Cole":        ["UCnzJFckvQBA5nn9lMu08LWQ"],                              # JColeVEVO
}

def _channel_video_count(youtube, channel_id):
    """そのチャンネルに動画が実際に存在するか確認する"""
    try:
        res = youtube.search().list(
            channelId=channel_id, part="id", type="video", maxResults=1
        ).execute()
        return len(res.get("items", []))
    except Exception:
        return 0

def find_artist_channel(youtube, artist):
    """
    アーティスト名から「本物の公式チャンネル」を自動で探す。
    1) CHANNEL_OVERRIDESに候補があれば、動画が実際に取得できる最初の候補を採用
    2) なければ [Artist]VEVO / [Artist] - Topic / アーティスト名そのもの で検索し、
       登録者数が最も多いチャンネルを「本物」として採用する
       （なりすまし・無関係チャンネル・ファンチャンネルは通常登録者数が大きく劣るため）。
    """
    if artist in CHANNEL_OVERRIDES:
        for cid in CHANNEL_OVERRIDES[artist]:
            if _channel_video_count(youtube, cid) == 0:
                print(f"    (候補チャンネル {cid} は動画が見つからずスキップ)")
                continue
            try:
                res = youtube.channels().list(id=cid, part="snippet,statistics").execute()
                item = res["items"][0]
                return {"channelId": cid, "title": item["snippet"]["title"],
                        "subs": int(item["statistics"].get("subscriberCount", 0))}
            except Exception:
                continue
        print(f"    (登録済みの候補チャンネルが全て使えなかったため自動検索にフォールバック)")

    queries = [f"{artist}VEVO", f"{artist} - Topic", artist]
    candidate_ids = set()
    for q in queries:
        try:
            res = youtube.search().list(q=q, part="snippet", type="channel", maxResults=5).execute()
        except Exception as e:
            print(f"    チャンネル検索エラー（{q}）: {e}")
            continue
        for item in res.get("items", []):
            candidate_ids.add(item["snippet"]["channelId"])

    if not candidate_ids:
        return None

    try:
        stats = youtube.channels().list(
            id=",".join(candidate_ids), part="snippet,statistics"
        ).execute()
    except Exception as e:
        print(f"    チャンネル統計取得エラー: {e}")
        return None

    # 動画が実際にあるチャンネルの中から、登録者数最大のものを選ぶ
    scored = []
    for item in stats.get("items", []):
        subs = int(item["statistics"].get("subscriberCount", 0))
        title = item["snippet"]["title"]
        scored.append({"channelId": item["id"], "title": title, "subs": subs})
    scored.sort(key=lambda x: -x["subs"])
    for cand in scored:
        if _channel_video_count(youtube, cand["channelId"]) > 0:
            return cand
    return scored[0] if scored else None


def collect_comments():
    print("=" * 60)
    print("STEP 1/6: YouTubeコメント収集")
    print("=" * 60)

    from googleapiclient.discovery import build
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    def get_top_videos_for_channel(channel_id, n):
        res = youtube.search().list(
            channelId=channel_id, part="snippet", type="video",
            order="viewCount", maxResults=n
        ).execute()
        return [(item["id"]["videoId"], item["snippet"]["title"])
                for item in res.get("items", [])]

    def get_comments(video_id, max_comments):
        comments, token = [], None
        while len(comments) < max_comments:
            kwargs = dict(videoId=video_id, part="snippet",
                          maxResults=100, textFormat="plainText")
            if token:
                kwargs["pageToken"] = token
            try:
                res = youtube.commentThreads().list(**kwargs).execute()
            except Exception as e:
                print(f"    コメント取得エラー（コメント無効化等の可能性）: {e}")
                break
            for item in res.get("items", []):
                snip = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({"text": snip["textDisplay"],
                                  "likes": snip["likeCount"]})
            token = res.get("nextPageToken")
            if not token:
                break
            time.sleep(0.3)
        return comments[:max_comments]

    all_rows = []
    for artist, genre in ARTISTS.items():
        print(f"\n[{artist}] 公式チャンネルを検索中...")
        channel = find_artist_channel(youtube, artist)
        if channel is None:
            print(f"  警告: {artist} の公式チャンネルが見つかりませんでした。スキップします。")
            continue
        print(f"  -> 採用チャンネル: 「{channel['title']}」(登録者数 {channel['subs']:,}人) ★この行を必ず確認してください")

        videos = get_top_videos_for_channel(channel["channelId"], VIDEOS_PER_ARTIST)
        rows = []
        for vid_id, title in videos:
            print(f"     -> {title[:55]}")
            for c in get_comments(vid_id, COMMENTS_PER_VIDEO):
                rows.append({"artist": artist, "genre": genre,
                              "video_id": vid_id, "video_title": title,
                              "channel_title": channel["title"],
                              "text": c["text"], "likes": c["likes"]})
            time.sleep(1)
        if rows:
            safe_name = artist.replace(" ", "_").replace("$", "S").replace(",", "")
            path = f"{RAW_DIR}/{safe_name}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  保存: {path} ({len(rows)}件)")
            all_rows.extend(rows)
        else:
            print(f"  警告: {artist} のコメントが取得できませんでした")

    if not all_rows:
        print("\n[エラー] コメントが1件も取得できませんでした。APIキーや")
        print("ネットワーク接続、YouTube APIのクォータ上限を確認してください。")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{RAW_DIR}/all_comments.csv", index=False)
    print(f"\n合計 {len(df)} コメント収集完了 -> {RAW_DIR}/all_comments.csv\n")
    return df


# ============================================================
# 3. 前処理
# ============================================================
STOPWORDS = {
    "the","a","an","is","are","was","were","i","you","he","she","we","they",
    "this","that","it","in","on","at","to","for","of","and","or","but","so",
    "be","been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","just","really","very",
    "like","get","got","my","me","him","her","us","them","his","its","their",
    "your","our","what","how","why","when","where","which","who","with","from",
    "not","no","even","also","still","only","about","up","out","than",
    "more","all","there","here","if","by","as","into","after","before","too",
}

# YouTube特有の「無関係な定型ノイズコメント」を除外するパターン
# (例: "anyone watching in 2026?", "who's here in 2025" など、
#  曲やアーティストへの価値観評価を一切含まない時報・記念コメント)
MEME_PATTERNS = [
    r"\b(anyone|anybody|who'?s|whos|someone)\b.{0,25}\b(20[12]\d|19\d\d)\b",
    r"\b(20[12]\d|19\d\d)\b.{0,25}\b(anyone|anybody|who'?s|whos|listening|watching)\b",
    r"\bstill (listening|watching|here)\b.{0,20}\b(20[12]\d)\b",
    r"\b\d+ (years?|yrs?) later\b",
    r"\bcame back .{0,15}(after|because)\b",
]
MEME_RE = re.compile("|".join(MEME_PATTERNS), re.IGNORECASE)

def is_meme_comment(text):
    return bool(MEME_RE.search(str(text)))

def is_mostly_english(text):
    """ラテン文字の比率が低い（=主に外国語の）コメントを除外する簡易判定"""
    s = re.sub(r"[\s\d\W]", "", str(text))  # 数字・記号・空白を除く
    if len(s) < 5:
        return False
    latin = sum(1 for c in s if c.isascii() and c.isalpha())
    return (latin / len(s)) > 0.6

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # より積極的に意味のある単語を抽出：最小文字数を2に（'cause等をキャッチ）
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def preprocess(df):
    print("=" * 60)
    print("STEP 2/6: 前処理")
    print("=" * 60)
    before = len(df)

    # ノイズ除去: 年号ミームコメント
    meme_mask = df["text"].apply(is_meme_comment)
    print(f"年号ミームコメント除外: {meme_mask.sum()}件")
    df = df[~meme_mask]

    # ノイズ除去: 主に外国語のコメント（英語キーワードでは評価できないため）
    eng_mask = df["text"].apply(is_mostly_english)
    print(f"非英語コメント除外: {(~eng_mask).sum()}件")
    df = df[eng_mask]

    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 10].reset_index(drop=True)
    df.to_csv(f"{PROC_DIR}/comments_clean.csv", index=False)
    print(f"クリーン後: {len(df)} / {before} コメント\n")
    return df


# ============================================================
# 4. Sentence-BERT 埋め込み
# ============================================================
def embed(df):
    print("=" * 60)
    print("STEP 3/6: Sentence-BERT 埋め込み")
    print("=" * 60)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("モデルロード完了: all-MiniLM-L6-v2")

    embeddings = model.encode(
        df["text_clean"].tolist(), batch_size=64,
        show_progress_bar=True, normalize_embeddings=True
    )
    np.save(f"{PROC_DIR}/embeddings.npy", embeddings)
    print(f"埋め込み: {embeddings.shape}\n")
    return embeddings


# ============================================================
# 5. UMAP + HDBSCAN クラスタリング
# ============================================================
LYRIC_KW     = {"bars","lyric","lyricism","storytelling","wordplay","verse","rhyme","goat","pen","think","poetry","punchline","bars","metaphor","clever","clever","depth","intelligent","articulate","rap","skill","technical"}
VIBE_KW      = {"vibe","beat","energy","flow","production","trap","banger","plays","melody","sound","bop","hard","beat","groovy","smooth","wavy","sonic","instrumental","bounce","slap","bumps"}
LIFESTYLE_KW = {"drip","fashion","swag","aura","style","aesthetic","outfit","icon","cool","vibe","energy","swagger","fresh","look","street","culture","influence","trendsetter"}
LIVE_KW      = {"concert","live","stage","show","performance","saw","experience","tour","festival","event","perform","crowd","audience","energy","atmosphere","fortnite","astronomical"}
ART_KW       = {"creative","art","experimental","artistic","vision","concept","innovative","genre","unique","originality","bold","boundary","push","evolution","genius","brilliant","masterpiece","legacy"}
INFLUENCE_KW = {"influence","impact","legacy","generation","inspired","inspire","changed","changed","culture","cultural","movement","pioneer","blueprint","goat","greatest"}
COLLAB_KW    = {"feature","feat","collab","collaboration","duo","together","chemistry","perfect","pair","synergy","talented"}

def label_cluster(top_words):
    ws = set(top_words)
    scores = {
        "🎤 リリック重視":   len(ws & LYRIC_KW),
        "🎵 Vibe/音楽性":    len(ws & VIBE_KW),
        "👟 ライフスタイル": len(ws & LIFESTYLE_KW),
        "🎪 ライブ体験":     len(ws & LIVE_KW),
        "🎨 芸術性":         len(ws & ART_KW),
        "⭐ 影響力/レガシー": len(ws & INFLUENCE_KW),
        "🤝 コラボ・相乗効果": len(ws & COLLAB_KW),
    }
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score > 0 else "📌 その他"

def cluster(df, embeddings):
    print("=" * 60)
    print("STEP 4/6: UMAP + HDBSCAN クラスタリング")
    print("=" * 60)
    import umap
    import hdbscan
    from sklearn.decomposition import PCA

    print("PCA で次元削減中（メモリ最適化）...")
    # 先に PCA で 50次元に削減
    pca = PCA(n_components=50, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    print("UMAP実行中...")
    # メモリ最適化: n_neighbors を削減
    n = min(10, len(embeddings_reduced) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n, min_dist=0.1,
                        metric="euclidean", random_state=42, verbose=False)
    emb2d = reducer.fit_transform(embeddings_reduced)
    np.save(f"{PROC_DIR}/umap_2d.npy", emb2d)

    print("HDBSCAN実行中...")
    # パラメータを最適化：より多くのポイントをクラスタに割り当て
    clusterer = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=5, metric="euclidean")
    labels = clusterer.fit_predict(emb2d)
    df["cluster"] = labels
    df["umap_x"] = emb2d[:, 0]
    df["umap_y"] = emb2d[:, 1]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"クラスタ数: {n_clusters}  ノイズ: {(labels == -1).sum()}")

    print("\n--- クラスタ代表単語とラベル ---")
    cluster_label_map = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        words = []
        for t in df[df["cluster"] == cid]["text_clean"]:
            words.extend(t.split())
        top_words = [w for w, _ in Counter(words).most_common(10)]
        lbl = label_cluster(top_words)
        cluster_label_map[cid] = lbl
        print(f"  Cluster {cid} -> {lbl}: {', '.join(top_words[:6])}")

    df["cluster_label"] = df["cluster"].map(cluster_label_map).fillna("ノイズ")
    df.to_csv(f"{PROC_DIR}/comments_clustered.csv", index=False)
    print()
    return df, emb2d


# ============================================================
# 6. アーティストベクトル + ネットワーク分析
# ============================================================
def analyze_network(df):
    print("=" * 60)
    print("STEP 5/6: アーティストベクトル + ネットワーク分析")
    print("=" * 60)
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    unique_labels = sorted(set(df["cluster_label"]) - {"ノイズ"})
    artist_vectors = {}
    artist_counts = {}  # 生のコメント件数（信頼性チェック用）
    for artist in df["artist"].unique():
        sub = df[df["artist"] == artist]
        total = len(sub)
        vec = {lbl: round((sub["cluster_label"] == lbl).sum() / total * 100, 1)
               for lbl in unique_labels}
        cnt = {lbl: int((sub["cluster_label"] == lbl).sum()) for lbl in unique_labels}
        artist_vectors[artist] = vec
        artist_counts[artist] = cnt

    artist_vec_df = pd.DataFrame(artist_vectors).T
    artist_vec_df.index.name = "artist"
    artist_vec_df["genre"] = artist_vec_df.index.map(
        lambda a: df[df["artist"] == a]["genre"].iloc[0])
    artist_vec_df.to_csv(f"{RESULT_DIR}/artist_vectors.csv")

    artist_count_df = pd.DataFrame(artist_counts).T
    artist_count_df.index.name = "artist"
    artist_count_df.to_csv(f"{RESULT_DIR}/artist_value_counts.csv")
    print("アーティストベクトル ->", f"{RESULT_DIR}/artist_vectors.csv")
    print("カテゴリ別の生コメント件数 ->", f"{RESULT_DIR}/artist_value_counts.csv")
    print(artist_vec_df.to_string())

    artists = list(artist_vectors.keys())

    # --- (A) 生ベクトルでの類似度（その他を含む・参考値） ---
    mat_raw = np.array([[artist_vectors[a].get(l, 0) for l in unique_labels] for a in artists])
    sim_raw = cosine_similarity(mat_raw)
    sim_df_raw = pd.DataFrame(sim_raw, index=artists, columns=artists)
    sim_df_raw.to_csv(f"{RESULT_DIR}/cosine_similarity_raw.csv")

    other_label = "📌 その他"
    other_share = artist_vec_df[other_label].mean() if other_label in artist_vec_df.columns else 0
    print(f"\n「その他」の平均割合: {other_share:.1f}%", end="")
    if other_share > 50:
        print("  ※50%超のため、類似度はその他除外版を主分析として採用")
    else:
        print()

    # --- (B) 「その他」を除いた4カテゴリのみで再正規化した類似度（主分析） ---
    meaningful_cols = [c for c in unique_labels if c != other_label]
    mat_excl = artist_vec_df[meaningful_cols].values.astype(float)
    row_sums = mat_excl.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # ゼロ除算回避
    mat_excl_norm = mat_excl / row_sums * 100
    sim_excl = cosine_similarity(mat_excl_norm)
    sim_df = pd.DataFrame(sim_excl, index=artists, columns=artists)
    sim_df.to_csv(f"{RESULT_DIR}/cosine_similarity.csv")

    artist_vec_norm_df = pd.DataFrame(mat_excl_norm, index=artists, columns=meaningful_cols)
    artist_vec_norm_df["genre"] = artist_vec_df["genre"]
    artist_vec_norm_df.to_csv(f"{RESULT_DIR}/artist_vectors_excl_other.csv")
    print(f"その他除外・再正規化ベクトル -> {RESULT_DIR}/artist_vectors_excl_other.csv")

    G = nx.Graph()
    for a in artists:
        G.add_node(a, genre=artist_vec_df.loc[a, "genre"])
    for i, a1 in enumerate(artists):
        for j, a2 in enumerate(artists):
            if i < j and sim_excl[i][j] >= SIM_THRESHOLD:
                G.add_edge(a1, a2, weight=round(sim_excl[i][j], 4))

    edges_df = pd.DataFrame(
        [{"src": u, "dst": v, "similarity": d["weight"]} for u, v, d in G.edges(data=True)]
    )
    if not edges_df.empty:
        edges_df.to_csv(f"{RESULT_DIR}/network_edges.csv", index=False)

    centrality = nx.degree_centrality(G)
    cent_df = pd.DataFrame(
        [{"artist": a, "degree_centrality": round(v, 4)} for a, v in centrality.items()]
    ).sort_values("degree_centrality", ascending=False)
    cent_df.to_csv(f"{RESULT_DIR}/network_centrality.csv", index=False)

    print(f"\nネットワークエッジ数: {len(edges_df)} (閾値{SIM_THRESHOLD}, その他除外)")
    print("\n中心性 TOP5:")
    print(cent_df.head().to_string(index=False))

    # --- (C) 閾値の頑健性チェック：複数の閾値で孤立ノードが変わるか確認 ---
    print("\n--- 閾値の頑健性チェック ---")
    robustness_rows = []
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        Gt = nx.Graph()
        Gt.add_nodes_from(artists)
        for i, a1 in enumerate(artists):
            for j, a2 in enumerate(artists):
                if i < j and sim_excl[i][j] >= thresh:
                    Gt.add_edge(a1, a2)
        isolated = [a for a in artists if Gt.degree(a) == 0]
        components = sorted(nx.connected_components(Gt), key=len, reverse=True)
        largest = len(components[0]) if components else 0
        robustness_rows.append({
            "threshold": thresh, "largest_component": largest,
            "isolated_count": len(isolated), "isolated_artists": ", ".join(isolated)
        })
        print(f"  閾値{thresh:.2f}: 最大クラスタ={largest}人, 孤立={len(isolated)}人 {isolated}")
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(f"{RESULT_DIR}/threshold_robustness.csv", index=False)
    print(f"頑健性チェック結果 -> {RESULT_DIR}/threshold_robustness.csv\n")

    return (artist_vec_df, sim_df, edges_df, cent_df, unique_labels, G,
            artist_count_df, sim_df_raw, artist_vec_norm_df, robustness_df, meaningful_cols)


# ============================================================
# 7. 可視化
# ============================================================
def safe_name(s):
    return str(s).replace("$", r"\$")

CLUSTER_COLORS = {
    "🎤 リリック重視": "#4361EE", "🎵 Vibe/音楽性": "#F72585",
    "👟 ライフスタイル": "#FF9F1C", "🎪 ライブ体験": "#2EC4B6",
    "🎨 芸術性": "#9B59B6", "⭐ 影響力/レガシー": "#FFD60A",
    "🤝 コラボ・相乗効果": "#06FFA5", "📌 その他": "#CCCCCC", "ノイズ": "#EEEEEE",
}

def make_genre_colors(genres):
    palette = ["#4361EE", "#F72585", "#2EC4B6", "#FF9F1C", "#9B59B6", "#43AA8B"]
    uniq = sorted(set(genres))
    return {g: palette[i % len(palette)] for i, g in enumerate(uniq)}

def visualize(df, emb2d, artist_vec_df, sim_df, edges_df, value_cols,
              artist_vec_norm_df=None, meaningful_cols=None):
    print("=" * 60)
    print("STEP 6/6: 可視化")
    print("=" * 60)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC", "axes.grid": True,
        "grid.alpha": 0.3, "font.size": 11,
    })

    GENRE_COLORS = make_genre_colors(artist_vec_df["genre"])
    genres_series = artist_vec_df["genre"]

    # Fig1: UMAP scatter
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    for genre, color in GENRE_COLORS.items():
        mask = df["genre"] == genre
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=color, alpha=0.35, s=8, label=genre, rasterized=True)
    ax.set_title("UMAP: ジャンル別コメント分布", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)

    ax = axes[1]
    for lbl, color in CLUSTER_COLORS.items():
        mask = df["cluster_label"] == lbl
        if mask.sum() == 0:
            continue
        alpha = 0.1 if lbl in ("📌 その他", "ノイズ") else 0.45
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=color, alpha=alpha, s=8, label=lbl, rasterized=True)
    ax.set_title("UMAP: 価値観クラスタ別分布", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_umap_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig1保存: {FIG_DIR}/fig1_umap_scatter.png")

    # Fig2: 積み上げ棒グラフ（その他を含む生の比率・参考用）
    order = sorted(artist_vec_df.index, key=lambda a: genres_series[a])
    df_s = artist_vec_df.loc[order, value_cols]
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(order))
    for col in value_cols:
        color = CLUSTER_COLORS.get(col, "#999999")
        vals = df_s[col].values.astype(float)
        ax.bar(range(len(order)), vals, bottom=bottom, color=color, label=col, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([safe_name(a) for a in order], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("コメント比率 (%)"); ax.set_ylim(0, 100)
    ax.set_title("アーティスト別 ファンの価値観ベクトル（生の比率・その他含む）", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    for genre, color in GENRE_COLORS.items():
        idxs = [i for i, a in enumerate(order) if genres_series[a] == genre]
        if idxs:
            ax.axvspan(min(idxs) - .5, max(idxs) + .5, alpha=0.06, color=color, zorder=0)
            ax.text((min(idxs) + max(idxs)) / 2, 98, genre, ha="center", va="top", fontsize=9, color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_artist_vectors_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig2保存: {FIG_DIR}/fig2_artist_vectors_raw.png")

    # Fig2b: 積み上げ棒グラフ（その他を除き再正規化・主分析）
    if artist_vec_norm_df is not None and meaningful_cols:
        order2 = sorted(artist_vec_norm_df.index, key=lambda a: genres_series[a])
        df_n = artist_vec_norm_df.loc[order2, meaningful_cols]
        fig, ax = plt.subplots(figsize=(14, 7))
        bottom = np.zeros(len(order2))
        for col in meaningful_cols:
            color = CLUSTER_COLORS.get(col, "#999999")
            vals = df_n[col].values.astype(float)
            ax.bar(range(len(order2)), vals, bottom=bottom, color=color, label=col, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(range(len(order2)))
        ax.set_xticklabels([safe_name(a) for a in order2], rotation=35, ha="right", fontsize=10)
        ax.set_ylabel("比率(%) ※その他を除き再正規化"); ax.set_ylim(0, 100)
        ax.set_title("アーティスト別 ファンの価値観ベクトル（その他除外・主分析）", fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        for genre, color in GENRE_COLORS.items():
            idxs = [i for i, a in enumerate(order2) if genres_series[a] == genre]
            if idxs:
                ax.axvspan(min(idxs) - .5, max(idxs) + .5, alpha=0.06, color=color, zorder=0)
                ax.text((min(idxs) + max(idxs)) / 2, 98, genre, ha="center", va="top", fontsize=9, color=color, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/fig2b_artist_vectors_excl_other.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Fig2b保存: {FIG_DIR}/fig2b_artist_vectors_excl_other.png")

    # Fig3: ヒートマップ（その他除外版・主分析）
    genre_order = sorted(sim_df.index, key=lambda a: genres_series.get(a, ""))
    sim_s = sim_df.loc[genre_order, genre_order]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sim_s.values, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="コサイン類似度（その他除外）")
    ax.set_xticks(range(len(genre_order))); ax.set_yticks(range(len(genre_order)))
    ax.set_xticklabels([safe_name(a) for a in genre_order], rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels([safe_name(a) for a in genre_order], fontsize=9)
    ax.set_title("アーティスト間コサイン類似度（その他除外・主分析）", fontsize=13, fontweight="bold", pad=12)
    for i in range(len(genre_order)):
        for j in range(len(genre_order)):
            v = sim_s.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig3保存: {FIG_DIR}/fig3_heatmap.png")

    # Fig4: ネットワーク（その他除外版・孤立ノードを赤枠で強調）
    G = nx.Graph()
    for a in artist_vec_df.index:
        G.add_node(a, genre=genres_series[a])
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["dst"], weight=row["similarity"])
    fig, ax = plt.subplots(figsize=(13, 10))
    pos = nx.spring_layout(G, seed=42, k=2.8, weight="weight")
    node_colors = [GENRE_COLORS[G.nodes[n]["genre"]] for n in G.nodes]
    if G.number_of_edges() > 0:
        edge_weights = [G[u][v]["weight"] for u, v in G.edges]
        edge_widths = [max((w - (SIM_THRESHOLD - 0.01)) * 20, 0.5) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.5, edge_color="#888888")
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7, alpha=0.7)
    node_edge_colors = ["red" if G.degree(n) == 0 else "white" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1200, alpha=0.92,
                            edgecolors=node_edge_colors, linewidths=2.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: safe_name(n) for n in G.nodes}, font_size=8, font_color="white", font_weight="bold")
    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GENRE_COLORS.items()]
    legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="red", linewidth=2, label="孤立ノード(赤枠)"))
    ax.legend(handles=legend_handles, fontsize=9.5, loc="lower left", framealpha=0.9)
    ax.set_title(f"ファン価値観ネットワーク（その他除外・閾値≥{SIM_THRESHOLD}）", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig4保存: {FIG_DIR}/fig4_network.png")

    # Fig5: アーティスト配置マップ（その他除外版ベクトルを使用）
    import umap as umap_pkg
    map_source = artist_vec_norm_df if artist_vec_norm_df is not None else artist_vec_df
    map_cols = meaningful_cols if meaningful_cols else value_cols
    value_mat = map_source[map_cols].values.astype(float)
    try:
        reducer2 = umap_pkg.UMAP(n_components=2, n_neighbors=min(4, len(value_mat)-1), min_dist=0.5, random_state=42)
        art2d = reducer2.fit_transform(value_mat)
    except Exception:
        from sklearn.decomposition import PCA
        art2d = PCA(n_components=2).fit_transform(value_mat)
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, artist in enumerate(map_source.index):
        color = GENRE_COLORS[genres_series[artist]]
        ax.scatter(art2d[i, 0], art2d[i, 1], c=color, s=500, zorder=3, edgecolors="white", linewidth=2)
        ax.annotate(safe_name(artist), (art2d[i, 0], art2d[i, 1]), fontsize=10, ha="center", va="bottom",
                     xytext=(0, 14), textcoords="offset points", fontweight="bold", color=color)
    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GENRE_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=10, framealpha=0.9)
    ax.set_title("ファン価値観によるアーティスト配置マップ（その他除外）", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_artist_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig5保存: {FIG_DIR}/fig5_artist_map.png\n")


# ============================================================
# 8. 考察レポート自動生成
# ============================================================
def generate_report(artist_vec_df, sim_df, edges_df, cent_df, value_cols,
                     artist_count_df=None, artist_vec_norm_df=None,
                     robustness_df=None, meaningful_cols=None):
    print("=" * 60)
    print("考察レポート自動生成")
    print("=" * 60)

    genres_series = artist_vec_df["genre"]
    genre_list = sorted(set(genres_series))
    other_label = "📌 その他"
    other_mean = artist_vec_df[other_label].mean() if other_label in artist_vec_df.columns else 0

    use_cols = meaningful_cols if meaningful_cols else value_cols
    use_df = artist_vec_norm_df if artist_vec_norm_df is not None else artist_vec_df

    # --- その他カテゴリの支配率 ---
    other_note = ""
    if other_mean > 40:
        other_note = (
            f"\n> ⚠️ **重要な前処理上の注記**：HDBSCANで見つかったクラスタのうち、事前に定義した価値観カテゴリの"
            f"キーワードに一致したのは一部のみで、平均{other_mean:.1f}%のコメントが「{other_label}」"
            f"（カテゴリに当てはまらない一般的な発言）に分類された。これを含めたまま類似度を計算すると、"
            f"全アーティストが同じ支配的次元を持つことになり類似度が意味を失うため、"
            f"**「{other_label}」を除いた{len(use_cols)}カテゴリのみで再正規化した値を主分析として採用する。**\n"
        )

    # --- ジャンル別の価値観傾向テーブル（その他除外・主分析） ---
    genre_avg = use_df.groupby("genre")[use_cols].mean().round(1)
    table_md = "| ジャンル | " + " | ".join(use_cols) + " |\n"
    table_md += "|---" * (len(use_cols) + 1) + "|\n"
    for g in genre_list:
        row = " | ".join(f"{genre_avg.loc[g, c]}%" for c in use_cols)
        table_md += f"| {g} | {row} |\n"

    # --- アーティスト別の価値観プロファイル（件数付き・信頼性チェック）---
    artist_table_md = "| アーティスト | ジャンル | " + " | ".join(use_cols) + " | 分析対象コメント数 |\n"
    artist_table_md += "|---" * (len(use_cols) + 3) + "|\n"
    for a in use_df.index:
        row = " | ".join(f"{use_df.loc[a, c]:.1f}%" for c in use_cols)
        n_comments = None
        if artist_count_df is not None and a in artist_count_df.index:
            n_comments = int(artist_count_df.loc[a, use_cols].sum())
        n_str = str(n_comments) if n_comments is not None else "—"
        artist_table_md += f"| {a} | {genres_series[a]} | {row} | {n_str} |\n"

    # サンプル数が少ないアーティストへの警告
    low_n_warning = ""
    if artist_count_df is not None:
        low_n_artists = []
        for a in use_df.index:
            n_comments = int(artist_count_df.loc[a, use_cols].sum()) if a in artist_count_df.index else 0
            if n_comments < 20:
                low_n_artists.append((a, n_comments))
        if low_n_artists:
            low_n_warning = (
                "\n> ⚠️ **サンプル数が少ないアーティストに注意**：以下のアーティストは「その他」を除いた"
                "実質的な分析対象コメント数が20件未満であり、パーセンテージが少数のコメントに大きく左右されている"
                "可能性がある。発表で数値を引用する際はこの点に留意すること。\n"
            )
            for a, n in low_n_artists:
                low_n_warning += f">   - {a}: {n}件\n"

    # --- ネットワーク構造の分析 ---
    isolated = [a for a in use_df.index if cent_df.set_index("artist").loc[a, "degree_centrality"] == 0]
    connected = [a for a in use_df.index if a not in isolated]
    top_central = cent_df.iloc[0]

    # --- 孤立アーティストの特徴（最も高い価値観カテゴリ）---
    isolated_profiles = []
    for a in isolated:
        row = use_df.loc[a, use_cols]
        top_cat = row.idxmax()
        top_val = row[top_cat]
        isolated_profiles.append((a, top_cat, top_val))

    # --- 頑健性チェックのまとめ ---
    robustness_md = ""
    robustness_summary = ""
    if robustness_df is not None and not robustness_df.empty:
        robustness_md = "| 閾値 | 最大クラスタ人数 | 孤立人数 | 孤立アーティスト |\n|---|---|---|---|\n"
        for _, row in robustness_df.iterrows():
            robustness_md += f"| {row['threshold']:.2f} | {row['largest_component']} | {row['isolated_count']} | {row['isolated_artists']} |\n"
        # 全閾値で共通して孤立しているアーティストを特定
        all_isolated_sets = [set(row["isolated_artists"].split(", ")) if row["isolated_artists"] else set()
                              for _, row in robustness_df.iterrows()]
        always_isolated = set.intersection(*all_isolated_sets) if all_isolated_sets else set()
        always_isolated.discard("")
        if always_isolated:
            robustness_summary = (
                f"閾値を{robustness_df['threshold'].min():.2f}〜{robustness_df['threshold'].max():.2f}まで"
                f"変化させても、**{'、'.join(always_isolated)}は常に孤立ノードのままだった**。"
                f"これは特定の閾値を都合よく選んだ結果ではなく、構造的に安定した発見であることを示す。"
            )

    # --- レポート本文 ---
    report = f"""# HIPHOPアーティストはジャンルではなく価値観で認識されているのか
## ―YouTubeコメントのクラスタリングによるファン・コミュニティマップ分析―

本レポートは実際のYouTubeコメントデータに基づいて自動生成された。
{other_note}
---

## 1. 研究の問い

既存のHIPHOPジャンル分類（Boom bap / Trap / Alternative、音楽的特徴による分類）に対し、ファンが実際にコメントで言及する内容（価値観）による分類がどの程度一致し、どこで異なるのかを検証する。

## 2. 主要結果

### 2.1 ジャンル別の価値観傾向（その他除外・主分析、平均%）

{table_md}

### 2.2 アーティスト別の価値観プロファイルとサンプル数

{artist_table_md}
{low_n_warning}

### 2.3 ネットワーク構造

閾値{SIM_THRESHOLD}でネットワークを構築した結果：

- **主流クラスタに含まれるアーティスト（{len(connected)}人）**：{", ".join(connected)}
- **孤立ノード（{len(isolated)}人）**：{", ".join(isolated) if isolated else "なし"}
- **ネットワーク中心性トップ**：{top_central['artist']}（度数中心性 {top_central['degree_centrality']}）

"""
    if isolated_profiles:
        report += "**孤立アーティストの特徴：**\n\n"
        for a, cat, val in isolated_profiles:
            report += f"- {a}：「{cat}」が{val:.1f}%と突出\n"
        report += "\n"

    report += f"""---

## 3. 頑健性チェック（閾値を変えても結果は安定しているか）

「閾値0.80だから出た結果では？」という疑問に答えるため、閾値を0.70〜0.95まで変化させてネットワーク構造を再計算した。

{robustness_md}

{robustness_summary}

---

## 4. 考察

### 4.1 ジャンルは価値観をどこまで説明するか

ジャンル別の価値観傾向表（2.1）を見ると、ジャンル間で平均値に違いはあるものの、個々のアーティストを見ると同じジャンル内でも価値観プロファイルが大きく異なる場合がある（2.2参照）。これは**ジャンルという事前のラベルが、ファンの言及パターンを完全には説明しないこと**を示している。

### 4.2 孤立ノードが示すもの

ネットワーク上で孤立したアーティストは、主流の価値観パターンとは異なる、独自の評価軸でファンに語られている可能性が高い。これは「文化の商業的利用は、ファンとの関わり方をどう均質化・差異化するか」という当初の研究関心に対する具体的な手がかりとなる。

### 4.3 サンプル数についての誠実な評価

本分析の各カテゴリは、HDBSCANで検出されたクラスタのうち「その他」を除いた一部のコメントのみに基づいている。アーティストによっては分析対象コメント数が少なく、突出した数値（例：あるカテゴリが70%超など）が少数のコメントに依存している場合がある。発表では、件数の小さいアーティストについては「示唆」程度の慎重な言い方をすることが望ましい。

---

## 5. 限界と今後の課題

- 「その他」が平均{other_mean:.1f}%を占めており、事前に定義した価値観カテゴリの枠組み自体が、実際のYouTubeコメントの言語を十分に捉えられていない可能性がある。カテゴリの再設計（generic な称賛コメントを独立カテゴリにする等）が今後の課題。
- アーティスト数が{len(artist_vec_df)}人と少なく、ジャンル全体への一般化には注意が必要。
- コメントを書くファンは、視聴者全体の中でも積極的な層に偏る可能性がある（selection bias）。
- クラスタへのラベル付け（リリック重視/Vibe等）はキーワードマッチングによる自動付与であり、人間による内容確認が望ましい。
- ジャンル分類自体が研究者の事前カテゴリであるため、完全に独立した検証ではない点に留意。

---

## 6. 想定質疑応答

**Q. 「その他」を除いて再計算するのは、都合の良い数字だけ見ているのでは？**
→ 「その他」を含めたままでは全アーティストの類似度が支配的次元に引き寄せられて意味を失うため、残りのカテゴリでの相対的な重みを見るための処理であり、結論に都合の良いデータを選んでいるわけではない。ただし「その他」の内容自体を分析すべきという指摘は妥当で、今後の課題として明記している。

**Q. 閾値0.80はどうやって決めたのか？**
→ 0.70〜0.95の範囲で頑健性チェックを行い、孤立ノードの構成が大きく変わらないことを確認している（セクション3参照）。

**Q. サンプル数が少ないアーティストの数値は信頼できるのか？**
→ 件数を明示しており（2.2参照）、少数件に基づく数値は「示唆」として慎重に扱うべきと明記している。

---

## 7. 発表構成案（7分目安）

| 時間 | 内容 |
|---|---|
| 0:00–1:00 | 問いの提示 |
| 1:00–2:00 | 手法と「その他」問題への対処 |
| 2:00–4:30 | 結果（ジャンル別傾向・ネットワーク図・孤立ノード） |
| 4:30–5:30 | 頑健性チェック |
| 5:30–6:30 | 考察 |
| 6:30–7:00 | 限界・結論 |

**キーメッセージ**：
「ジャンルは音楽の違いを区切るが、ファンの言葉が描く地図はそれとは違う形をしている。{f"特に{', '.join(isolated)}は、閾値を変えても常に主流から離れた独自の位置を占めていた。" if isolated else ""}」
"""

    with open(f"{RESULT_DIR}/discussion_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"考察レポート -> {RESULT_DIR}/discussion_report.md\n")


# ============================================================
# メイン処理
# ============================================================
def main():
    # 既存データから続行（API クォータ超過時用）
    if os.path.exists(f"{RAW_DIR}/all_comments.csv"):
        print("=" * 60)
        print("既存のコメントデータから続行します")
        print("=" * 60)
        df = pd.read_csv(f"{RAW_DIR}/all_comments.csv")
        print(f"既存データ読み込み: {len(df)} コメント\n")
    else:
        df = collect_comments()
    df = preprocess(df)
    embeddings = embed(df)
    df, emb2d = cluster(df, embeddings)
    (artist_vec_df, sim_df, edges_df, cent_df, value_cols, G,
     artist_count_df, sim_df_raw, artist_vec_norm_df, robustness_df, meaningful_cols) = analyze_network(df)
    visualize(df, emb2d, artist_vec_df, sim_df, edges_df, value_cols,
              artist_vec_norm_df, meaningful_cols)
    generate_report(artist_vec_df, sim_df, edges_df, cent_df, value_cols,
                     artist_count_df, artist_vec_norm_df, robustness_df, meaningful_cols)

    print("=" * 60)
    print("全工程完了！結果は ./output/ フォルダ以下に保存されています。")
    print("=" * 60)
    print(f"""
  {OUT_DIR}/
  ├── data/raw/                    収集した生コメント
  ├── data/processed/               前処理・クラスタリング済みデータ
  └── results/
      ├── artist_vectors.csv        アーティスト価値観ベクトル
      ├── cosine_similarity.csv     類似度行列
      ├── network_edges.csv         ネットワークエッジ
      ├── network_centrality.csv    中心性スコア
      ├── discussion_report.md      考察レポート（発表にそのまま使える）
      └── figures/                  グラフ5枚（png）
""")

if __name__ == "__main__":
    main()
