"""
engagement_metrics.py

コメントデータ（data/processed/{beef_folder}_comments_clean.csv）と
動画メタデータ（data/processed/all_video_metadata.csv）から熱量指標を計算する。

コメントデータから算出（期間別）:
  reply_rate        = reply_count合計 / コメント数（コメント1件あたりの平均返信数）
  like_rate         = like_count合計 / コメント数（コメント1件あたりの平均いいね数）
  avg_length        = コメントの平均文字数
  caps_ratio        = 大文字のみの単語（2文字以上）を含むコメントの割合
  exclamation_ratio = 感嘆符(!)を含むコメントの割合

動画メタデータと組み合わせて算出（動画別）:
  comment_per_view = comment_count / view_count
  like_per_view    = like_count / view_count

出力:
  data/processed/engagement_by_period.csv
  data/processed/engagement_by_video.csv

使い方:
  python engagement_metrics.py
"""

import csv
import re
from pathlib import Path

import pandas as pd

BEEF_ROOT = Path(__file__).resolve().parents[1]
BEEFS_CSV = BEEF_ROOT / "config" / "beefs.csv"
METADATA_CSV = BEEF_ROOT / "data" / "processed" / "all_video_metadata.csv"
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"

CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")


def load_beefs():
    with open(BEEFS_CSV, newline="", encoding="utf-8") as f:
        return {r["beef_id"]: r for r in csv.DictReader(f)}


def has_caps_word(text):
    return bool(CAPS_WORD_RE.search(text))


def engagement_by_period():
    beefs = load_beefs()
    rows = []

    for beef_id, beef_row in beefs.items():
        in_path = DATA_PROCESSED / f"{beef_row['folder_name']}_comments_clean.csv"
        if not in_path.exists():
            print(f"[{beef_id}] 前処理済みファイルが見つかりません: {in_path}（スキップ）")
            continue

        # period列には補助beefで"NA"という文字列値が入るため、欠損値として解釈されないよう
        # keep_default_na=Falseにし、空文字のみを欠損として扱う
        df = pd.read_csv(in_path, keep_default_na=False, na_values=[""])
        if df.empty:
            continue

        df["text"] = df["text"].fillna("")
        periods = ["PRE", "PEAK", "POST"] if beef_row["beef_role"] == "main" else sorted(df["period"].unique())

        for period in periods:
            sub = df[df["period"] == period]
            n = len(sub)
            if n == 0:
                rows.append({
                    "beef_id": beef_id, "period": period, "n_comments": 0,
                    "reply_rate": None, "like_rate": None, "avg_length": None,
                    "caps_ratio": None, "exclamation_ratio": None,
                })
                continue

            rows.append({
                "beef_id": beef_id,
                "period": period,
                "n_comments": n,
                "reply_rate": round(sub["reply_count"].sum() / n, 4),
                "like_rate": round(sub["like_count"].sum() / n, 4),
                "avg_length": round(sub["text"].str.len().mean(), 2),
                "caps_ratio": round(sub["text"].apply(has_caps_word).mean(), 4),
                "exclamation_ratio": round(sub["text"].str.contains("!", regex=False).mean(), 4),
            })

    out_path = DATA_PROCESSED / "engagement_by_period.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"期間別エンゲージメント指標 -> {out_path}")


def engagement_by_video():
    if not METADATA_CSV.exists():
        print(f"エラー: {METADATA_CSV} が見つかりません。先に fetch_video_stats.py を実行してください。")
        return

    df = pd.read_csv(METADATA_CSV)
    for col in ["view_count", "comment_count", "like_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["view_count"] > 0].copy()
    df["comment_per_view"] = df["comment_count"] / df["view_count"]
    df["like_per_view"] = df["like_count"] / df["view_count"]

    out_cols = ["video_id", "song_name", "artist", "beef_id", "view_count",
                "comment_count", "like_count", "comment_per_view", "like_per_view"]
    out_path = DATA_PROCESSED / "engagement_by_video.csv"
    df[out_cols].sort_values("comment_per_view", ascending=False).to_csv(out_path, index=False)
    print(f"動画別エンゲージメント指標 -> {out_path}")


def main():
    engagement_by_period()
    engagement_by_video()


if __name__ == "__main__":
    main()
