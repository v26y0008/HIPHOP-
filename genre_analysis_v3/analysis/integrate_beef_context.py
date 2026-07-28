"""
integrate_beef_context.py (genre_v3)

beef_analysis の v2 コメントデータ（comments_clean_v2.csv）から、
ビーフ当事者8アーティスト（Kendrick Lamar, Drake, Pusha T, Eminem, MGK,
Lil Wayne, Megan Thee Stallion, Nicki Minaj）分のコメントを取り出し、
genre_v3のスキーマに合わせて data/processed/beef_context_clean.csv に保存する。

位置づけ:
  - これらのアーティストは boom_bap/trap/alternative のジャンル分類には含めず、
    genre="beef_context" という独立カテゴリとして扱う
  - track_category は catalog/beef/post の3種類が揃っている（v2の元設計のまま）
    - catalog/postのみのジャンル比較クラスタリングに使う場合は track_category
      でフィルタすること
    - beef_id等のビーフ固有メタデータは lyric_axis_beef.py 側で
      songs_v2.csv と再結合して利用する

Kendrick Lamar は「genre_v3で新規収集したboom_bapの5曲+3曲」と「v2由来のbeef_context
3曲（catalog=HUMBLE, beef=Not Like Us, post=tv off）」の両方に登場する。
これは重複ではなく、同一アーティストを異なる文脈（純粋ジャンル比較 vs ビーフ文脈）で
別々に保持する意図的な設計。

使い方:
  python integrate_beef_context.py
"""

from pathlib import Path

import pandas as pd

GENRE_ROOT = Path(__file__).resolve().parents[1]
BEEF_ROOT = GENRE_ROOT.parent / "beef_analysis"
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_PATH = GENRE_ROOT / "data" / "processed" / "beef_context_clean.csv"

BEEF_ARTISTS = [
    "Kendrick Lamar", "Drake", "Pusha T", "Eminem", "MGK",
    "Lil Wayne", "Megan Thee Stallion", "Nicki Minaj",
]


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df = df[df["artist_name"].isin(BEEF_ARTISTS)].copy()
    df["genre"] = "beef_context"

    keep_cols = [
        "comment_id", "video_id", "key", "genre", "artist_name", "track_category",
        "release_date", "days_since_release", "song_name",
        "published_at", "text", "like_count", "reply_count", "text_clean",
        "beef_id", "beef_name", "beef_type", "artist_side",
    ]
    df = df[keep_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"抽出件数: {len(df):,} -> {OUT_PATH}")
    print("\nartist_name x track_category 別件数:")
    print(df.groupby(["artist_name", "track_category"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
