"""
fetch_video_stats.py (v2 設計)

config/songs_v2.csv に記入された video_id（空欄以外）について videos.list を実行し、
再生回数・コメント数・いいね数などのメタデータを
data/processed/all_video_metadata_v2.csv に保存する。

B3(Drake vs Meek Mill)はbeefカテゴリが欠損しユーザー判断で分析から除外済みのため、
songs_v2.csv側でvideo_idを空欄化してある行は自動的にスキップされる。

使い方:
  python fetch_video_stats.py
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
BEEF_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = BEEF_ROOT / "config" / "songs_v2.csv"
OUTPUT_CSV = BEEF_ROOT / "data" / "processed" / "all_video_metadata_v2.csv"
QUOTA_LOG = BEEF_ROOT / "logs" / "api_quota_usage.log"

FIELDNAMES = [
    "video_id", "key", "song_name", "artist_name", "beef_id", "track_category",
    "release_date", "collected_at", "view_count", "comment_count", "like_count",
    "channel_title", "video_title",
]

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")


def fetch_stats(youtube, video_id):
    resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    item = items[0]
    snippet = item["snippet"]
    stats = item.get("statistics", {})
    return {
        "view_count": stats.get("viewCount", "NA"),
        "comment_count": stats.get("commentCount", "NA"),
        "like_count": stats.get("likeCount", "NA"),
        "channel_title": snippet.get("channelTitle", ""),
        "video_title": snippet.get("title", ""),
    }


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: 環境変数 YOUTUBE_API_KEY が設定されていません。")
        sys.exit(1)

    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    todo_rows = [r for r in rows if r["video_id"].strip()]
    skipped = len(rows) - len(todo_rows)
    print(f"対象: {len(todo_rows)}件（video_id未設定/B3除外でスキップ: {skipped}件）")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    collected_at = datetime.now(timezone.utc).isoformat()

    output_rows = []
    n_ok, n_error = 0, 0
    units_used = 0

    for row in todo_rows:
        video_id = row["video_id"].strip()
        print(f"取得中: [{row['key']}] {row['artist_name']} - {row['song_name']} (video_id={video_id})")
        try:
            stats = fetch_stats(youtube, video_id)
            units_used += 1
        except Exception as e:
            print(f"  !! APIエラー: {e}")
            n_error += 1
            continue

        if stats is None:
            print("  !! 動画が見つかりませんでした。")
            n_error += 1
            continue

        print(f"  再生数: {stats['view_count']} / コメント数: {stats['comment_count']} / いいね数: {stats['like_count']}")

        output_rows.append({
            "video_id": video_id,
            "key": row["key"],
            "song_name": row["song_name"],
            "artist_name": row["artist_name"],
            "beef_id": row["beef_id"],
            "track_category": row["track_category"],
            "release_date": row["release_date"],
            "collected_at": collected_at,
            "view_count": stats["view_count"],
            "comment_count": stats["comment_count"],
            "like_count": stats["like_count"],
            "channel_title": stats["channel_title"],
            "video_title": stats["video_title"],
        })
        n_ok += 1
        time.sleep(0.2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUOTA_LOG, "a", encoding="utf-8") as f:
        f.write(f"{collected_at} fetch_video_stats.py(v2) videos.list_calls={units_used} units_used={units_used}\n")

    print(f"\n完了: {n_ok}件を {OUTPUT_CSV} に保存（collected_at={collected_at}）。エラー={n_error}")
    print(f"APIクォータ消費: {units_used} units")


if __name__ == "__main__":
    main()
