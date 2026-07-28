"""
check_video_metadata.py

本収集の前に必ず実行する確認ステップ。
config/songs.csv に記入された video_id を videos.list で1件ずつ引き、
タイトル・チャンネル名・再生回数などを取得してログに残す。

【目的】
過去に「アーティスト名の文字列検索」でチャンネルを特定した結果、
Nas の検索が Nelly / Lil Nas X の動画に誤って一致した事故があった。
本スクリプトは動画IDベースでの確認を徹底し、
チャンネル名・タイトルに artist 名が含まれているかを機械的にチェックすることで、
本収集（fetch_video_stats.py / fetch_comments.py）に進む前に人間が目視確認できるようにする。

使い方:
  python check_video_metadata.py
  → config/songs.csv を読み込み、logs/video_metadata_check.log に結果を出力する
  （video_id が REPLACE_ME のままの行はスキップされる）

事前準備:
  1. リポジトリ直下の .env に YOUTUBE_API_KEY を設定
  2. config/songs.csv の video_id 列を、実際に公式チャンネルで確認した動画IDに書き換える
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
SONGS_CSV = BEEF_ROOT / "config" / "songs.csv"
LOG_PATH = BEEF_ROOT / "logs" / "video_metadata_check.log"

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")


def name_key_for(artist):
    """アーティスト名からチェック用の主要トークンを1つ取り出す（既存スクリプトと同じ簡易方式）"""
    return artist.split(",")[0].split("$")[0].strip().split()[0].lower()


def check_video(youtube, video_id):
    try:
        resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    except Exception as e:
        return None, f"APIエラー: {e}"

    items = resp.get("items", [])
    if not items:
        return None, "動画が見つかりません（video_idを確認してください）"

    item = items[0]
    snippet = item["snippet"]
    stats = item.get("statistics", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "view_count": stats.get("viewCount", "NA"),
        "comment_count": stats.get("commentCount", "NA"),
        "like_count": stats.get("likeCount", "NA"),
    }, None


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("エラー: 環境変数 YOUTUBE_API_KEY が設定されていません（リポジトリ直下の .env を確認してください）。")
        sys.exit(1)

    if not SONGS_CSV.exists():
        print(f"エラー: {SONGS_CSV} が見つかりません。")
        sys.exit(1)

    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    placeholder_rows = [r for r in rows if r["video_id"].strip() == "REPLACE_ME"]
    todo_rows = [r for r in rows if r["video_id"].strip() != "REPLACE_ME"]

    print(f"config/songs.csv: 全{len(rows)}件中、video_id未設定（REPLACE_ME）が{len(placeholder_rows)}件あります。")
    if todo_rows:
        print(f"確認対象: {len(todo_rows)}件\n")
    else:
        print("確認対象の動画IDが0件です。songs.csv の video_id 列を記入してから再実行してください。")
        return

    youtube = build("youtube", "v3", developerKey=API_KEY)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_lines = [f"=== video metadata check: {datetime.now(timezone.utc).isoformat()} ==="]
    n_ok, n_warn, n_error = 0, 0, 0

    for row in todo_rows:
        artist = row["artist"]
        song = row["song_name"]
        video_id = row["video_id"].strip()

        print(f"確認中: [{row['beef_id']}] {artist} - {song} (video_id={video_id})")
        meta, error = check_video(youtube, video_id)

        if error:
            print(f"  !! エラー: {error}")
            log_lines.append(f"ERROR beef_id={row['beef_id']} song={song} artist={artist} video_id={video_id} -- {error}")
            n_error += 1
            time.sleep(0.2)
            continue

        key = name_key_for(artist)
        haystack = f"{meta['channel_title']} {meta['title']}".lower()
        name_match = key in haystack

        status = "OK" if name_match else "WARN"
        if name_match:
            n_ok += 1
        else:
            n_warn += 1
            print(f"  !! 警告: チャンネル名 '{meta['channel_title']}' / タイトル '{meta['title']}' に "
                  f"'{key}' が含まれていません。想定アーティストと一致するか目視確認してください。")

        print(f"  タイトル: {meta['title']}")
        print(f"  チャンネル: {meta['channel_title']}")
        print(f"  公開日: {meta['published_at']}")
        print(f"  再生数: {meta['view_count']} / コメント数: {meta['comment_count']} / いいね数: {meta['like_count']}\n")

        log_lines.append(
            f"{status} beef_id={row['beef_id']} beef_role={row['beef_role']} song={song} artist={artist} "
            f"video_id={video_id} title=\"{meta['title']}\" channel=\"{meta['channel_title']}\" "
            f"published_at={meta['published_at']} view_count={meta['view_count']} "
            f"comment_count={meta['comment_count']} like_count={meta['like_count']}"
        )

        time.sleep(0.2)  # レート制限対策

    log_lines.append(f"=== summary: OK={n_ok} WARN={n_warn} ERROR={n_error} SKIPPED(placeholder)={len(placeholder_rows)} ===\n")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"完了: OK={n_ok} / 警告={n_warn} / エラー={n_error} / 未設定スキップ={len(placeholder_rows)}")
    print(f"詳細ログ: {LOG_PATH}")
    if n_warn or n_error:
        print("\n警告・エラーのあった行は、本収集（fetch_video_stats.py）に進む前に必ず目視で確認してください。")


if __name__ == "__main__":
    main()
