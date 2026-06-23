"""
fetch_youtube_comments.py

YouTube Data API v3 を使って、指定した動画からコメントを取得するスクリプト。
このファイル単体で実行可能（ご自身のPC / Google Colab等、ネットワーク制限のない環境で実行してください）。

【重要】旧データで発生した不具合への対応
旧パイプラインでは「アーティスト名でチャンネルを検索」する方式を取っていたため、
"Nas" の検索が誤って Nelly / Lil Nas X / Nelly Furtado の動画にマッチしてしまうという
事故が起きた。これを避けるため、本スクリプトは「動画IDを直接指定する」方式のみを採用する。

事前準備:
  1. pip install google-api-python-client pandas
  2. https://console.cloud.google.com/ でプロジェクトを作成し、YouTube Data API v3 を有効化
  3. APIキーを発行し、下記 API_KEY に設定（あるいは環境変数 YOUTUBE_API_KEY を使用）
  4. VIDEO_LIST の各動画IDを、実際に公式チャンネルのMVであることを確認した上で記入する
     （動画IDはYouTube URLの "watch?v=" の後の11文字の部分）

使い方:
  python fetch_youtube_comments.py
  → all_comments_new.csv が生成される
"""

import os
import time
import csv
import sys

try:
    from googleapiclient.discovery import build
except ImportError:
    print("先に次を実行してください: pip install google-api-python-client")
    sys.exit(1)

# ============================================================
# 設定
# ============================================================

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")

N_COMMENTS_PER_VIDEO = 400  # 動画1本あたりの取得コメント数（上限）

# ------------------------------------------------------------
# 動画リスト：(アーティスト名, ジャンル, 動画ID, 動画タイトルの目印)
#
# 【要対応】下記の video_id は記入例のプレースホルダーです。
# 実際にYouTubeで各アーティストの公式チャンネルを開き、
# ライブ/コンサート/インタビュー以外の公式MV・公式Audioを3本ずつ選んで、
# URLの watch?v= の後ろ11文字をここに入力してください。
#
# 例: https://www.youtube.com/watch?v=PBwbqsfDpKE
#                                      ^^^^^^^^^^^ ←これが動画ID
# ------------------------------------------------------------
VIDEO_LIST = [
    # ===== Boom bap / Lyric =====
    ("Nas",            "Boom bap", "RvVfgvHucRY", "I Can"),
    ("Nas",            "Boom bap", "VC4ORS5n9Hg", "Nas Is Like"),
    ("Nas",            "Boom bap", "e5PnuIRnJW8", "The World Is Yours"),

    ("Joey Bada$$",    "Boom bap", "51e1gIkzHgk", "Waves"),
    ("Joey Bada$$",    "Boom bap", "RLnA25dVzrQ", "Devastated"),
    ("Joey Bada$$",    "Boom bap", "yRfQGXFRr30", "Christ Conscious"),

    ("Kendrick Lamar",    "Boom bap", "H58vbez_m4E", "Not Like Us"),
    ("Kendrick Lamar",    "Boom bap", "tvTRZJ-4EyI", "HUMBLE."),
    ("Kendrick Lamar",    "Boom bap", "Z-48u_uWMHY", "Alright"),

    ("J. Cole",      "Boom bap", "xvZqHgFz51I", "Middle Child"),
    ("J. Cole",      "Boom bap", "qAlF70MdfwU", "No Role Modelz"),
    ("J. Cole",      "Boom bap", "pWgZhh3Fb-w", "ATM"),

    # ===== Trap =====
    ("Future",         "Trap", "6OxmafNPn3o", "LIL DEMON"),
    ("Future",         "Trap", "xvZqHgFz51I", "Mask Off"),
    ("Future",         "Trap", "l0U7SxXHkPY", "Life Is Good"),

    ("Travis Scott",          "Trap", "tfSS1e3kYeo", "HIGHEST IN THE ROOM"),
    ("Travis Scott",          "Trap", "6ONRf7h3Mdk", "SICKO MODE"),
    ("Travis Scott",          "Trap", "X7aF3nZOS98", "I KNOW ?"),

    ("Playboi Carti",     "Trap", "KnumAWWWgUE", "Sky"),
    ("Playboi Carti",     "Trap", "VcRc2DHHhoM", "EVIL J0RDAN"),
    ("Playboi Carti",     "Trap", "oCveByMXd_0", "Magnolia"),

    ("Young Thug",      "Trap", "dwWs7ZnGekc", "Relationship"),
    ("Young Thug",      "Trap", "p0x6FEDV-ig", "Check"),
    ("Young Thug",      "Trap", "nGt_JGHYEO4", "Lifestyle"),

    # ===== Alternative =====
    ("Tyler, The Creator", "Alternative", "ljHdccyQbT4", "SUGAR ON MY TONGUE"),
    ("Tyler, The Creator", "Alternative", "TGgcC5xg9YI", "SEE YOU AGAIN"),
    ("Tyler, The Creator", "Alternative", "OxlJLz9M8hQ", "Tamale"),

    ("JPEGMAFIA",      "Alternative", "gIn4ZoMgwp8", "babygirl"),
    ("JPEGMAFIA",      "Alternative", "SpH83KzVKDc", "SIN MIEDO"),
    ("JPEGMAFIA",      "Alternative", "2CGFU1lBdCI", "HAZARD DUTY PAY!"),

    ("A$AP Rocky", "Alternative", "Kbj2Zss-5GY", "Praise The Lord"),
    ("A$AP Rocky", "Alternative", "Gx4JEBwVlXo", "L$D"),
    ("A$AP Rocky", "Alternative", "F6VfsJ7LAlE", "Fashion Killa"),

    ("Mac Miller",    "Alternative", "SsKT0s5J8ko", "Self Care"),
    ("Mac Miller",    "Alternative", "aIHF7u9Wwiw", "Good News"),
    ("Mac Miller",    "Alternative", "6bMmhKz6KXg", "Knock Knock"),
]

OUTPUT_CSV = "all_comments_new.csv"


def get_video_title(youtube, video_id):
    """動画のタイトルとチャンネル名を取得（収集ログ・検証用）"""
    try:
        resp = youtube.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return None, None
        snippet = items[0]["snippet"]
        return snippet["title"], snippet["channelTitle"]
    except Exception as e:
        print(f"  動画情報取得エラー ({video_id}): {e}")
        return None, None


def fetch_comments(youtube, video_id, max_results=400):
    """指定した動画のトップレベルコメントを取得する"""
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            ).execute()
        except Exception as e:
            print(f"  コメント取得エラー ({video_id}): {e}")
            break

        for item in resp.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": top_comment.get("textDisplay", ""),
                "likes": top_comment.get("likeCount", 0),
            })

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.2)  # レート制限対策

    return comments[:max_results]


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: API_KEY が設定されていません。")
        print("環境変数 YOUTUBE_API_KEY を設定するか、スクリプト内の API_KEY を直接書き換えてください。")
        sys.exit(1)

    placeholder_count = sum(1 for _, _, vid, _ in VIDEO_LIST if vid.startswith("REPLACE_ME"))
    if placeholder_count > 0:
        print(f"警告: VIDEO_LIST に未設定の動画ID（REPLACE_ME）が {placeholder_count} 件あります。")
        print("実行前に、各アーティストの実際の動画IDに置き換えてください。")
        print("このまま続行すると、未設定の行はスキップされます。\n")

    youtube = build("youtube", "v3", developerKey=API_KEY)

    rows = []
    for artist, genre, video_id, _ in VIDEO_LIST:
        if video_id.startswith("REPLACE_ME"):
            continue

        print(f"取得中: {artist} ({genre}) - video_id={video_id}")
        title, channel_title = get_video_title(youtube, video_id)

        if title is None:
            print("  -> 動画が見つかりませんでした。video_idを確認してください。")
            continue

        # ---- 検証ステップ：チャンネル名にアーティスト名の主要部分が
        #      含まれているかを軽くチェックし、Nasのような誤データ混入を防ぐ ----
        name_key = artist.split(",")[0].split("$")[0].strip().split()[0].lower()
        if name_key not in channel_title.lower() and name_key not in title.lower():
            print(f"  !! 警告: チャンネル名 '{channel_title}' / タイトル '{title}' に "
                  f"'{name_key}' が含まれていません。意図したアーティストか確認してください。")

        print(f"  動画: {title}  (チャンネル: {channel_title})")

        comments = fetch_comments(youtube, video_id, max_results=N_COMMENTS_PER_VIDEO)
        print(f"  -> {len(comments)} 件のコメントを取得")

        for c in comments:
            rows.append({
                "artist": artist,
                "genre": genre,
                "video_id": video_id,
                "video_title": title,
                "channel_title": channel_title,
                "text": c["text"],
                "likes": c["likes"],
            })

        time.sleep(0.5)  # レート制限対策

    if not rows:
        print("\n取得できたコメントが0件でした。VIDEO_LISTの動画IDを設定してから再実行してください。")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "genre", "video_id", "video_title", "channel_title", "text", "likes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n完了: {len(rows)} 件のコメントを {OUTPUT_CSV} に保存しました。")

    # 簡易サマリー
    from collections import Counter
    artist_counts = Counter(r["artist"] for r in rows)
    print("\nアーティスト別取得件数:")
    for artist, count in artist_counts.items():
        print(f"  {artist}: {count}件")


if __name__ == "__main__":
    main()
