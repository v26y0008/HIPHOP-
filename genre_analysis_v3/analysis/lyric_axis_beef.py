"""
lyric_axis_beef.py (genre_v3)

「歌詞軸(lyric axis)」と「vibe軸」という2つの語彙カテゴリを研究者定義し、
コメントがどちらの観点でアーティストを語っているかを定量化する。
これをジャンル間比較・ビーフ文脈との統合分析に使う。

LYRIC_VOCAB: 歌詞の内容・技巧・メッセージ性に注目したコメントで使われがちな語
             （バース、韻、パンチライン、メッセージ性など）
VIBE_VOCAB : 音・雰囲気・プロダクションに注目したコメントで使われがちな語
             （ビート、雰囲気、ノリ、メロディーなど）

これらは筆者が定義した研究用の語彙リストであり、正解の分類ではなく操作的定義である
（conflict_vocab_analysis.py の CONFLICT_VOCAB と同じ位置づけ）。

分析内容:
  1. genre別 (boom_bap/trap/alternative/beef_context) の lyric_rate / vibe_rate
     を比較 → 「boom_bapは歌詞、trap/alternativeはvibeで語られやすい」という
     よくある批評言説を実データで検証
  2. beef_context アーティストについて、track_category (catalog/beef/post) 別の
     lyric_rate / vibe_rate を比較 → ビーフ中は歌詞/ディス内容への言及が増えるか

使い方:
  python lyric_axis_beef.py
"""

import re
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

GENRE_ROOT = Path(__file__).resolve().parents[1]
GENRE_CLEAN = GENRE_ROOT / "data" / "processed" / "comments_clean_genre_v3.csv"
BEEF_CONTEXT_CLEAN = GENRE_ROOT / "data" / "processed" / "beef_context_clean.csv"
OUT_DIR = GENRE_ROOT / "data" / "processed"

LYRIC_VOCAB = [
    "lyric", "lyrics", "lyricist", "bar", "bars", "verse", "verses", "rhyme", "rhymes",
    "rhyming", "punchline", "punchlines", "wordplay", "metaphor", "metaphors", "bar for bar",
    "storytelling", "story telling", "message", "meaning", "meaningful", "deep lyrics",
    "wrote this", "writing", "poetic", "poetry", "clever", "double entendre", "flow",
    "raps about", "spitting", "spits", "real hip hop", "substance", "lyricism",
]

VIBE_VOCAB = [
    "vibe", "vibes", "beat", "beats", "production", "produced", "instrumental", "melody",
    "melodic", "melodies", "mood", "atmosphere", "aesthetic", "sound", "sounds like",
    "bangs", "goes hard", "hits different", "chill", "energy", "catchy", "replay",
    "replaying", "on repeat", "bassline", "bass", "drop", "the beat is", "production value",
    "hook", "chorus", "sample", "samples",
]


def compile_pattern(words):
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


LYRIC_PATTERN = compile_pattern(LYRIC_VOCAB)
VIBE_PATTERN = compile_pattern(VIBE_VOCAB)


def has_lyric_vocab(text):
    return bool(LYRIC_PATTERN.search(str(text)))


def has_vibe_vocab(text):
    return bool(VIBE_PATTERN.search(str(text)))


def load_corpus():
    frames = []
    if GENRE_CLEAN.exists():
        df1 = pd.read_csv(GENRE_CLEAN, keep_default_na=False, na_values=[""])
        frames.append(df1[["comment_id", "genre", "artist_name", "track_category", "text_clean"]])
    if BEEF_CONTEXT_CLEAN.exists():
        df2 = pd.read_csv(BEEF_CONTEXT_CLEAN, keep_default_na=False, na_values=[""])
        frames.append(df2[["comment_id", "genre", "artist_name", "track_category", "text_clean"]])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def summarize_rates(df, group_cols):
    grp = df.groupby(group_cols).agg(
        n=("comment_id", "size"),
        lyric_rate=("has_lyric", "mean"),
        vibe_rate=("has_vibe", "mean"),
    ).reset_index()
    grp["lyric_rate"] = (grp["lyric_rate"] * 100).round(2)
    grp["vibe_rate"] = (grp["vibe_rate"] * 100).round(2)
    return grp


def main():
    df = load_corpus()
    if df is None:
        print("エラー: 入力データが見つかりません。preprocess_genre.py / integrate_beef_context.py を先に実行してください。")
        return

    df["has_lyric"] = df["text_clean"].apply(has_lyric_vocab)
    df["has_vibe"] = df["text_clean"].apply(has_vibe_vocab)

    print(f"総コメント数: {len(df):,}")
    print(f"lyric語彙を含む割合: {df['has_lyric'].mean() * 100:.2f}%")
    print(f"vibe語彙を含む割合: {df['has_vibe'].mean() * 100:.2f}%")

    # 1) genre別
    by_genre = summarize_rates(df, ["genre"])
    print("\n=== genre別 lyric_rate / vibe_rate ===")
    print(by_genre)

    genre_only = df[df["genre"] != "beef_context"]
    def cramers_v(ct):
        chi2, p, _, _ = chi2_contingency(ct)
        n = ct.values.sum()
        v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
        return chi2, p, v

    ct_genre = pd.crosstab(genre_only["genre"], genre_only["has_lyric"])
    if ct_genre.shape[0] > 1 and ct_genre.shape[1] > 1:
        chi2_l, p_l, v_l = cramers_v(ct_genre)
    else:
        chi2_l, p_l, v_l = None, None, None
    ct_genre_v = pd.crosstab(genre_only["genre"], genre_only["has_vibe"])
    if ct_genre_v.shape[0] > 1 and ct_genre_v.shape[1] > 1:
        chi2_v, p_v, v_v = cramers_v(ct_genre_v)
    else:
        chi2_v, p_v, v_v = None, None, None
    print(f"\ngenre x lyric_vocab カイ二乗検定: chi2={chi2_l:.2f}, p={p_l:.4g}, Cramer's V={v_l:.3f}")
    print(f"genre x vibe_vocab カイ二乗検定: chi2={chi2_v:.2f}, p={p_v:.4g}, Cramer's V={v_v:.3f}")

    # 2) beef_contextアーティストのtrack_category別
    beef_only = df[df["genre"] == "beef_context"]
    by_beef_cat = summarize_rates(beef_only, ["artist_name", "track_category"])
    print("\n=== beef_context アーティスト x track_category別 lyric_rate / vibe_rate ===")
    print(by_beef_cat)

    by_cat_overall = summarize_rates(beef_only, ["track_category"])
    print("\n=== beef_context 全体 track_category別（アーティスト横断）===")
    print(by_cat_overall)

    ct_cat = pd.crosstab(beef_only["track_category"], beef_only["has_lyric"])
    if ct_cat.shape[0] > 1 and ct_cat.shape[1] > 1:
        chi2_cat_l, p_cat_l, _, _ = chi2_contingency(ct_cat)
    else:
        chi2_cat_l, p_cat_l = None, None
    print(f"\ntrack_category x lyric_vocab (beef_context) カイ二乗検定: chi2={chi2_cat_l}, p={p_cat_l}")

    # --- 保存 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_genre.to_csv(OUT_DIR / "lyric_vibe_by_genre.csv", index=False)
    by_beef_cat.to_csv(OUT_DIR / "lyric_vibe_by_beef_artist_category.csv", index=False)
    by_cat_overall.to_csv(OUT_DIR / "lyric_vibe_by_beef_category_overall.csv", index=False)

    per_artist = summarize_rates(df, ["genre", "artist_name", "track_category"])
    per_artist.to_csv(OUT_DIR / "lyric_vibe_per_artist_category.csv", index=False)

    print(f"\n保存完了: {OUT_DIR}")


if __name__ == "__main__":
    main()
