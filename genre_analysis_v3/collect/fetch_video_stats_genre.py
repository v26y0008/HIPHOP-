"""
fetch_video_stats_genre.py (genre_v3)

config/songs_genre_v3.csv の全152本について videos.list をバッチ実行(50件ずつ)し、
再生回数・コメント数・いいね数などのメタデータを
data/processed/all_video_metadata_genre_v3.csv に保存する。

beef_analysis版と異なりバッチ取得（1件ずつではなく50件/リクエスト）でクォータを節約する。

使い方:
  python fetch_video_stats_genre.py
"""

import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from googleapiclient.discovery import build
except ImportError:
    print("先に次を実行してください: pip install google-api-python-client")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENRE_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = GENRE_ROOT / "config" / "songs_genre_v3.csv"
OUTPUT_CSV = GENRE_ROOT / "data" / "processed" / "all_video_metadata_genre_v3.csv"
QUOTA_LOG = GENRE_ROOT / "logs" / "api_quota_usage.log"

FIELDNAMES = [
    "video_id", "key", "song_name", "artist_name", "genre", "track_category",
    "release_date", "collected_at", "view_count", "comment_count", "like_count",
    "channel_title", "video_title",
]

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

API_KEY = os.environ.get("YOUTUBE_API_KEY_2", os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE"))


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: 環境変数 YOUTUBE_API_KEY_2 / YOUTUBE_API_KEY が設定されていません。")
        sys.exit(1)

    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"対象: {len(rows)}件")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    collected_at = datetime.now(timezone.utc).isoformat()

    info = {}
    units_used = 0
    all_ids = [r["video_id"].strip() for r in rows]
    for i in range(0, len(all_ids), 50):
        batch = all_ids[i:i + 50]
        resp = youtube.videos().list(part="snippet,statistics", id=",".join(batch)).execute()
        units_used += 1
        for item in resp.get("items", []):
            info[item["id"]] = item
        time.sleep(0.2)

    output_rows = []
    n_ok, n_error = 0, 0

    for row in rows:
        video_id = row["video_id"].strip()
        item = info.get(video_id)
        if item is None:
            print(f"  !! 動画が見つかりませんでした: [{row['key']}] {video_id}")
            n_error += 1
            continue

        snippet = item["snippet"]
        stats = item.get("statistics", {})
        output_rows.append({
            "video_id": video_id,
            "key": row["key"],
            "song_name": row["song_name"],
            "artist_name": row["artist_name"],
            "genre": row["genre"],
            "track_category": row["track_category"],
            "release_date": snippet.get("publishedAt", "")[:10],
            "collected_at": collected_at,
            "view_count": stats.get("viewCount", "NA"),
            "comment_count": stats.get("commentCount", "NA"),
            "like_count": stats.get("likeCount", "NA"),
            "channel_title": snippet.get("channelTitle", ""),
            "video_title": snippet.get("title", ""),
        })
        n_ok += 1

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUOTA_LOG, "a", encoding="utf-8") as f:
        f.write(f"{collected_at} fetch_video_stats_genre.py videos.list_calls={units_used} units_used={units_used}\n")

    print(f"\n完了: {n_ok}件を {OUTPUT_CSV} に保存（collected_at={collected_at}）。エラー={n_error}")
    print(f"APIクォータ消費: {units_used} units")


if __name__ == "__main__":
    main()
