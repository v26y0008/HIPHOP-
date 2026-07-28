"""
preprocess.py (v2 設計)

data/processed/all_comments_v2.csv を読み込み、テキストを前処理して
data/processed/comments_clean_v2.csv に保存する。

前処理内容:
  - URL の除去
  - メンション（@username）の除去
  - 小文字統一
  - 言語フィルタ：英語コメントのみ（langdetectで判定）
  - 4語未満の極端に短いコメントを除去

使い方:
  python preprocess.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
except ImportError:
    print("先に次を実行してください: pip install langdetect")
    sys.exit(1)

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "all_comments_v2.csv"
OUT_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
MIN_WORDS = 4


def clean_text(text):
    text = URL_RE.sub("", str(text))
    text = MENTION_RE.sub("", text)
    text = text.lower()
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
        print(f"エラー: {IN_PATH} が見つかりません。先に build_dataset.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    print(f"前処理前: {len(df):,}件")

    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.split().str.len() >= MIN_WORDS]
    df = df[df["text_clean"].apply(is_english)]

    print(f"前処理後: {len(df):,}件（除去率 {(1 - len(df) / len(pd.read_csv(IN_PATH))) * 100:.1f}%）")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"-> {OUT_PATH}")

    print("\nbeef_id x track_category 別件数（前処理後）:")
    print(df.groupby(["beef_id", "track_category"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
