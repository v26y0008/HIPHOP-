"""
preprocess_genre.py (genre_v3)

data/processed/all_comments_genre_v3.csv を読み込み、テキストを前処理して
data/processed/comments_clean_genre_v3.csv に保存する。

beef_analysis/analysis/preprocess.py と同じ基本前処理（URL/メンション除去、小文字化、
英語フィルタ、4語未満除去）に加えて、genre_v3独自の処理として

  - 固有名詞除去（アーティスト名・曲名）: クラスタリングが「誰の名前が出てくるか」
    だけで分離してしまうのを防ぐため（教授指摘への直接回答: ジャンルラベルを使わない
    クラスタリングが、実質的にアーティスト名の有無で分離しているだけにならないようにする）

を行う。

使い方:
  python preprocess_genre.py
"""

import re
import sys
import csv
from pathlib import Path

import pandas as pd

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
except ImportError:
    print("先に次を実行してください: pip install langdetect")
    sys.exit(1)

GENRE_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = GENRE_ROOT / "data" / "processed" / "all_comments_genre_v3.csv"
SONGS_CSV = GENRE_ROOT / "config" / "songs_genre_v3.csv"
OUT_PATH = GENRE_ROOT / "data" / "processed" / "comments_clean_genre_v3.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
MIN_WORDS = 4

# アーティスト名の略称・別表記（固有名詞除去用に追加で潰す語）
ARTIST_ALIASES = {
    "J. Cole": ["j cole", "jcole", "cole"],
    "Nas": ["nas", "nasir"],
    "Joey Bada$$": ["joey badass", "joey bada", "badass"],
    "Freddie Gibbs": ["freddie gibbs", "gibbs"],
    "Rapsody": ["rapsody"],
    "Travis Scott": ["travis scott", "travis", "la flame"],
    "Young Thug": ["young thug", "thugger"],
    "Future": ["future", "pluto", "hendrix"],
    "21 Savage": ["21 savage", "savage"],
    "Lil Baby": ["lil baby", "baby"],
    "Gunna": ["gunna", "wunna"],
    "Tyler, The Creator": ["tyler the creator", "tyler", "creator"],
    "Mac Miller": ["mac miller", "mac"],
    "JPEGMAFIA": ["jpegmafia", "jpeg", "peggy"],
    "Denzel Curry": ["denzel curry", "denzel", "zeltron"],
    "Childish Gambino": ["childish gambino", "gambino", "donald glover"],
    "Saba": ["saba"],
    "A$AP Rocky": ["asap rocky", "a ap rocky", "rocky", "flacko"],
    "Playboi Carti": ["playboi carti", "carti", "opium"],
}


def build_removal_patterns():
    names = set()
    for artist, aliases in ARTIST_ALIASES.items():
        names.add(artist.lower())
        names.update(aliases)

    if SONGS_CSV.exists():
        with open(SONGS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                song = row.get("song_name", "").strip()
                if song:
                    names.add(song.lower())

    names = [re.escape(n) for n in names if len(n) >= 3]
    names.sort(key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(names) + r")\b", flags=re.IGNORECASE)
    return pattern


def clean_text(text, name_pattern):
    text = URL_RE.sub("", str(text))
    text = MENTION_RE.sub("", text)
    text = text.lower()
    text = name_pattern.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_english(text):
    if not text:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に build_genre_dataset.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    n_before = len(df)
    print(f"前処理前: {n_before:,}件")

    name_pattern = build_removal_patterns()

    df["text_clean"] = df["text"].apply(lambda t: clean_text(t, name_pattern))
    df = df[df["text_clean"].str.split().str.len() >= MIN_WORDS]
    df = df[df["text_clean"].apply(is_english)]

    print(f"前処理後: {len(df):,}件（除去率 {(1 - len(df) / n_before) * 100:.1f}%）")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"-> {OUT_PATH}")

    print("\ngenre x artist_name x track_category 別件数（前処理後）:")
    print(df.groupby(["genre", "artist_name", "track_category"]).size())


if __name__ == "__main__":
    main()
