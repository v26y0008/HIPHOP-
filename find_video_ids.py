"""
find_video_ids.py

fetch_youtube_comments.py の VIDEO_LIST に入れる動画IDを探すための補助スクリプト。
アーティスト名で検索し、候補動画のタイトル・チャンネル名・video_idを一覧表示する。
表示された中から「公式チャンネルのMV」を目で確認して選び、
fetch_youtube_comments.py の VIDEO_LIST に書き込んでください。

使い方:
  python find_video_ids.py "Nas" "official video"
  python find_video_ids.py "Joey Bada$$" "official video"
"""

import os
import sys

try:
    from googleapiclient.discovery import build
except ImportError:
    print("先に次を実行してください: pip install google-api-python-client")
    sys.exit(1)

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")


def search_videos(youtube, query, max_results=10):
    resp = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        order="relevance",
    ).execute()

    results = []
    for item in resp.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        channel_title = item["snippet"]["channelTitle"]
        results.append((video_id, title, channel_title))
    return results


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: 環境変数 YOUTUBE_API_KEY を設定するか、スクリプト内のAPI_KEYを書き換えてください。")
        sys.exit(1)

    if len(sys.argv) < 2:
        print('使い方: python find_video_ids.py "アーティスト名" [追加キーワード]')
        sys.exit(1)

    artist = sys.argv[1]
    extra = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "official video"
    query = f"{artist} {extra}"

    youtube = build("youtube", "v3", developerKey=API_KEY)
    print(f"検索クエリ: {query}\n")

    results = search_videos(youtube, query)

    print(f"{'video_id':<14} {'チャンネル名':<30} タイトル")
    print("-" * 100)
    for video_id, title, channel_title in results:
        print(f"{video_id:<14} {channel_title:<30} {title}")

    print("\n上記の中から、公式チャンネル・公式MV/Audioのものを選び、")
    print("fetch_youtube_comments.py の VIDEO_LIST に video_id を書き込んでください。")
    print("（ライブ・コンサート・インタビュー・ファン投稿動画は除外してください）")


if __name__ == "__main__":
    main()
