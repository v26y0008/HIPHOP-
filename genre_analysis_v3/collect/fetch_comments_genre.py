"""
fetch_comments_genre.py (genre_v3)

config/songs_genre_v3.csv に記載された全動画(19アーティスト x 8本 = 152本)について
コメントを取得し、data/raw/{key}_comments.json に保存する。

beef_analysis/collect/fetch_comments.py と同じ設計:
  - order=time固定で最大 --max-comments 件（デフォルト1000件）
  - コメント欄無効(HttpError commentsDisabled)はエラーを記録してスキップ

使い方:
  python fetch_comments_genre.py                      # 全動画を処理
  python fetch_comments_genre.py --artist j_cole       # 特定アーティストのみ
  python fetch_comments_genre.py --key j_cole_catalog_01  # 特定の1本のみ
"""

import argparse
import csv
import json
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
    from googleapiclient.errors import HttpError
except ImportError:
    print("先に次を実行してください: pip install google-api-python-client")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENRE_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = GENRE_ROOT / "config" / "songs_genre_v3.csv"
DATA_RAW = GENRE_ROOT / "data" / "raw"
QUOTA_LOG = GENRE_ROOT / "logs" / "api_quota_usage.log"
SUMMARY_LOG = GENRE_ROOT / "logs" / "collection_summary.log"

MAX_COMMENTS_DEFAULT = 1000

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

API_KEY = os.environ.get("YOUTUBE_API_KEY_2", os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE"))


def load_songs():
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_comments(youtube, video_id, max_results):
    comments = []
    next_page_token = None
    quota_used = 0

    while len(comments) < max_results:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                order="time",
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()
            quota_used += 1
        except HttpError as e:
            return comments, quota_used, str(e)

        for item in resp.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "text": snip.get("textOriginal", ""),
                "published_at": snip.get("publishedAt", ""),
                "like_count": snip.get("likeCount", 0),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
            })

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.2)

    return comments, quota_used, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist", default=None, help="特定のartist_nameのみ処理")
    parser.add_argument("--key", default=None, help="特定のkeyのみ処理")
    parser.add_argument("--max-comments", type=int, default=MAX_COMMENTS_DEFAULT)
    parser.add_argument("--skip-existing", action="store_true", help="既に取得済み(JSONが存在)の動画をスキップ")
    args = parser.parse_args()

    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: 環境変数 YOUTUBE_API_KEY_2 / YOUTUBE_API_KEY が設定されていません。")
        sys.exit(1)

    songs = load_songs()
    if args.artist:
        songs = [s for s in songs if s["artist_name"] == args.artist]
    if args.key:
        songs = [s for s in songs if s["key"] == args.key]

    if args.skip_existing:
        songs = [s for s in songs if not (DATA_RAW / f"{s['key']}_comments.json").exists()]

    youtube = build("youtube", "v3", developerKey=API_KEY)
    run_ts = datetime.now(timezone.utc).isoformat()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    total_quota = 0
    n_ok, n_disabled, n_error = 0, 0, 0
    summary_rows = []

    for i, row in enumerate(songs, 1):
        key = row["key"]
        video_id = row["video_id"].strip()
        out_path = DATA_RAW / f"{key}_comments.json"

        print(f"[{i}/{len(songs)}] [収集中] {key} ({video_id}) genre={row['genre']} artist={row['artist_name']} category={row['track_category']}")
        comments, quota, error = fetch_comments(youtube, video_id, args.max_comments)
        total_quota += quota

        payload = {
            "key": key,
            "video_id": video_id,
            "genre": row["genre"],
            "artist_name": row["artist_name"],
            "track_category": row["track_category"],
            "collected_at": run_ts,
            "n_comments": len(comments),
            "error": error,
            "comments": comments,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if error:
            disabled = "commentsDisabled" in error
            if disabled:
                print(f"  !! コメント欄無効のためスキップ")
                n_disabled += 1
            else:
                print(f"  !! エラー: {error}")
                n_error += 1
        else:
            print(f"  -> {len(comments)}件保存 (quota使用: {quota})")
            n_ok += 1

        summary_rows.append(
            f"{run_ts} fetch_comments_genre.py key={key} artist={row['artist_name']} "
            f"category={row['track_category']} n_comments={len(comments)} quota={quota} error={error}"
        )
        time.sleep(0.3)

    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUOTA_LOG, "a", encoding="utf-8") as f:
        f.write(f"{run_ts} fetch_comments_genre.py commentThreads.list_calls={total_quota} units_used={total_quota}\n")
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")

    print(f"\n完了: OK={n_ok} / コメント欄無効={n_disabled} / エラー={n_error}")
    print(f"合計APIクォータ消費: {total_quota} units（commentThreads.list）")


if __name__ == "__main__":
    main()
