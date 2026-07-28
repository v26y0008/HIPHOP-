"""
name_patterns.py (genre_v3 共通モジュール)

genre_v3の19アーティスト + ビーフ文脈8アーティスト、および両方のsongs configの
曲名から固有名詞除去用の正規表現パターンを作る。

クラスタリングが「誰の名前/曲名が出てくるか」だけで分離してしまうのを防ぐため
（教授指摘への直接回答）、テキストクラスタリングに使う前段階で必ずこれを通す。
"""

import csv
import re
from pathlib import Path

GENRE_ROOT = Path(__file__).resolve().parents[1]
BEEF_ROOT = GENRE_ROOT.parent / "beef_analysis"

GENRE_SONGS_CSV = GENRE_ROOT / "config" / "songs_genre_v3.csv"
BEEF_SONGS_CSV = BEEF_ROOT / "config" / "songs_v2.csv"

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
    "Kendrick Lamar": ["kendrick lamar", "kendrick", "kdot", "k dot", "cornrow kenny"],
    "Drake": ["drake", "drizzy", "champagne papi", "aubrey"],
    "Pusha T": ["pusha t", "pusha", "king push"],
    "Eminem": ["eminem", "em", "slim shady", "marshall"],
    "MGK": ["mgk", "machine gun kelly"],
    "Lil Wayne": ["lil wayne", "wayne", "weezy", "tunechi"],
    "Megan Thee Stallion": ["megan thee stallion", "megan", "meg thee stallion", "hot girl meg"],
    "Nicki Minaj": ["nicki minaj", "nicki", "onika"],
}


def build_removal_pattern():
    names = set()
    for artist, aliases in ARTIST_ALIASES.items():
        names.add(artist.lower())
        names.update(aliases)

    for csv_path in (GENRE_SONGS_CSV, BEEF_SONGS_CSV):
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                song = row.get("song_name", "").strip()
                if song:
                    names.add(song.lower())

    names = [re.escape(n) for n in names if len(n) >= 3]
    names.sort(key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(names) + r")\b", flags=re.IGNORECASE)


WHITESPACE_RE = re.compile(r"\s+")


def strip_names(text, pattern):
    text = pattern.sub(" ", str(text).lower())
    return WHITESPACE_RE.sub(" ", text).strip()
