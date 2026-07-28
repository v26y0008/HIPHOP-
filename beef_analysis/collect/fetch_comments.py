"""
fetch_comments.py (v2 設計)

config/songs_v2.csv に記載された動画（video_idが空でない行）についてコメントを取得し、
data/raw/{key}_comments.json に保存する。

v1設計との違い:
  - publishedAtによる期間フィルタは行わない（動画の種類=track_categoryそのものが区分）
  - order=time固定で最大 --max-comments 件（デフォルト1500件）を取得するだけ
  - コメント欄が無効な動画（HttpError commentsDisabled）はエラーを記録してスキップする

使い方:
  python fetch_comments.py                  # 全曲を処理
  python fetch_comments.py --beef-id b1      # 特定beefのみ処理
  python fetch_comments.py --key b6_beef_nicki  # 特定の1曲のみ処理
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
BEEF_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = BEEF_ROOT / "config" / "songs_v2.csv"
DATA_RAW = BEEF_ROOT / "data" / "raw_v2"
QUOTA_LOG = BEEF_ROOT / "logs" / "api_quota_usage.log"
SUMMARY_LOG = BEEF_ROOT / "logs" / "collection_summary.log"

MAX_COMMENTS_DEFAULT = 1500

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")


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
    parser.add_argument("--beef-id", default=None, help="特定のbeef_idのみ処理（例: b1）")
    parser.add_argument("--key", default=None, help="特定のkeyのみ処理（例: b6_beef_nicki）")
    parser.add_argument("--max-comments", type=int, default=MAX_COMMENTS_DEFAULT)
    args = parser.parse_args()

    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: 環境変数 YOUTUBE_API_KEY が設定されていません（リポジトリ直下の .env を確認してください）。")
        sys.exit(1)

    songs = load_songs()
    if args.beef_id:
        songs = [s for s in songs if s["beef_id"] == args.beef_id]
    if args.key:
        songs = [s for s in songs if s["key"] == args.key]

    todo = [s for s in songs if s["video_id"].strip()]
    skipped_no_video = len(songs) - len(todo)
    if skipped_no_video:
        print(f"video_id未設定の行が{skipped_no_video}件あります（片側型ビーフ等。スキップ）")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    run_ts = datetime.now(timezone.utc).isoformat()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    total_quota = 0
    n_ok, n_disabled, n_error = 0, 0, 0
    summary_rows = []

    for row in todo:
        key = row["key"]
        video_id = row["video_id"].strip()
        out_path = DATA_RAW / f"{key}_comments.json"

        print(f"[収集中] {key} ({video_id}) beef={row['beef_id']} category={row['track_category']} artist={row['artist_side']}")
        comments, quota, error = fetch_comments(youtube, video_id, args.max_comments)
        total_quota += quota

        payload = {
            "key": key,
            "video_id": video_id,
            "beef_id": row["beef_id"],
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
            f"{run_ts} fetch_comments.py(v2) key={key} beef_id={row['beef_id']} "
            f"category={row['track_category']} n_comments={len(comments)} quota={quota} error={error}"
        )
        time.sleep(0.3)

    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUOTA_LOG, "a", encoding="utf-8") as f:
        f.write(f"{run_ts} fetch_comments.py(v2) commentThreads.list_calls={total_quota} units_used={total_quota}\n")
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")

    print(f"\n完了: OK={n_ok} / コメント欄無効={n_disabled} / エラー={n_error} / video_id未設定スキップ={skipped_no_video}")
    print(f"合計APIクォータ消費: {total_quota} units（commentThreads.list）")


if __name__ == "__main__":
    main()
