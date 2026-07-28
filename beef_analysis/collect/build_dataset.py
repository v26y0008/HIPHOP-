"""
build_dataset.py (v2 設計)

data/raw_v2/{key}_comments.json と config/songs_v2.csv を突き合わせて
1つの統合CSV（data/processed/all_comments_v2.csv）を作る。

各コメントに付与するメタデータ:
  beef_id, beef_name, beef_type, drake_involved,
  track_category (catalog/beef/post), artist_side, artist_name,
  release_date, days_since_release, video_id, song_name

使い方:
  python build_dataset.py
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

BEEF_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = BEEF_ROOT / "config" / "songs_v2.csv"
DATA_RAW = BEEF_ROOT / "data" / "raw_v2"
OUT_PATH = BEEF_ROOT / "data" / "processed" / "all_comments_v2.csv"

# B3(Drake vs Meek Mill)・B7(Ice Cube vs NWA)はいずれもbeefカテゴリが
# 両アーティストともコメント欄無効(403 commentsDisabled)で完全に欠損したため除外
EXCLUDED_BEEFS = {"b3", "b7"}

FIELDNAMES = [
    "comment_id", "video_id", "key", "beef_id", "beef_name", "beef_type",
    "drake_involved", "track_category", "artist_side", "artist_name",
    "release_date", "days_since_release", "song_name",
    "published_at", "text", "like_count", "reply_count",
]


def load_songs():
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def main():
    songs = load_songs()
    rows = []
    n_skipped_no_meta = 0
    n_skipped_disabled = 0

    for json_path in sorted(DATA_RAW.glob("*_comments.json")):
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)

        key = payload["key"]
        meta = songs.get(key)
        if meta is None:
            print(f"[警告] {key} のメタデータがsongs_v2.csvにありません（スキップ）")
            n_skipped_no_meta += 1
            continue

        if meta["beef_id"] in EXCLUDED_BEEFS:
            continue

        if payload.get("error"):
            print(f"[スキップ] {key}: 収集時にエラー（{payload['error'][:60]}...）")
            n_skipped_disabled += 1
            continue

        release_date = datetime.strptime(meta["release_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)

        for c in payload["comments"]:
            published_at = datetime.fromisoformat(c["published_at"].replace("Z", "+00:00"))
            days_since_release = (published_at - release_date).days

            rows.append({
                "comment_id": c["comment_id"],
                "video_id": c["video_id"],
                "key": key,
                "beef_id": meta["beef_id"],
                "beef_name": meta["beef_name"],
                "beef_type": meta["beef_type"],
                "drake_involved": meta["drake_involved"],
                "track_category": meta["track_category"],
                "artist_side": meta["artist_side"],
                "artist_name": meta["artist_name"],
                "release_date": meta["release_date"],
                "days_since_release": days_since_release,
                "song_name": meta["song_name"],
                "published_at": c["published_at"],
                "text": c["text"],
                "like_count": c["like_count"],
                "reply_count": c["reply_count"],
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"総コメント数: {len(rows):,} -> {OUT_PATH}")
    print(f"(メタデータ欠如でスキップ: {n_skipped_no_meta}件, 収集エラーでスキップ: {n_skipped_disabled}件)")

    # beef_id x track_category のクロス集計を表示
    counts = {}
    for r in rows:
        k = (r["beef_id"], r["track_category"])
        counts[k] = counts.get(k, 0) + 1
    beef_ids = sorted({r["beef_id"] for r in rows})
    categories = ["catalog", "beef", "post"]
    print(f"\n{'beef_id':<8}" + "".join(f"{c:>10}" for c in categories))
    for bid in beef_ids:
        print(f"{bid:<8}" + "".join(f"{counts.get((bid, c), 0):>10}" for c in categories))


if __name__ == "__main__":
    main()
