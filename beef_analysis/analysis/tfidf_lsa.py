"""
tfidf_lsa.py (v2 設計)

TF-IDFでtrack_category別（catalog/beef/post）の上位語彙を抽出し、
「ビーフ曲でどんな語彙が支配的になるか」を比較する。
さらにビーフタイプ別（rap_battle vs contract_breakdown）のビーフ曲上位語彙も比較する。

使い方:
  python tfidf_lsa.py
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"

N_TOP = 20


def top_words_per_group(df, group_col, n_top=N_TOP):
    results = {}
    for group in sorted(df[group_col].dropna().unique()):
        texts = df[df[group_col] == group]["text_clean"].dropna().tolist()
        if len(texts) < 10:
            continue
        vectorizer = TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2), stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        scores = tfidf.mean(axis=0).A1
        words = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:n_top]
        results[group] = [(words[i], float(scores[i])) for i in top_idx]
    return results


def save_top_words(results, out_path):
    rows = []
    for group, words in results.items():
        for rank, (word, score) in enumerate(words, start=1):
            rows.append({"group": group, "rank": rank, "word": word, "tfidf_score": score})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に preprocess.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])

    # ============================================================
    # 分析1: track_category別（全ビーフ込み）の上位語
    # ============================================================
    print("=== track_category別 TF-IDF上位語（全ビーフ込み） ===")
    top_by_category = top_words_per_group(df, "track_category")
    for cat, words in top_by_category.items():
        print(f"\n[{cat}]")
        print(", ".join(w for w, s in words[:15]))
    save_top_words(top_by_category, OUT_DIR / "tfidf_top_words_by_category.csv")

    # ============================================================
    # 分析2: ビーフタイプ別（ビーフ曲のみ）の上位語
    # 「ラップバトル型 vs 契約破局型」で語彙が違うか
    # ============================================================
    print("\n=== ビーフタイプ別（beefカテゴリのみ）TF-IDF上位語 ===")
    beef_only = df[df["track_category"] == "beef"]
    top_by_type = top_words_per_group(beef_only, "beef_type")
    for btype, words in top_by_type.items():
        print(f"\n[{btype}]")
        print(", ".join(w for w, s in words[:15]))
    save_top_words(top_by_type, OUT_DIR / "tfidf_top_words_by_beef_type.csv")

    # ============================================================
    # 分析3: ビーフ別（ビーフ曲のみ）の上位語
    # ============================================================
    print("\n=== ビーフ別（beefカテゴリのみ）TF-IDF上位語 ===")
    top_by_beef = top_words_per_group(beef_only, "beef_id")
    for bid, words in top_by_beef.items():
        print(f"\n[{bid}]")
        print(", ".join(w for w, s in words[:15]))
    save_top_words(top_by_beef, OUT_DIR / "tfidf_top_words_by_beef.csv")

    print(f"\n出力:\n  {OUT_DIR / 'tfidf_top_words_by_category.csv'}\n"
          f"  {OUT_DIR / 'tfidf_top_words_by_beef_type.csv'}\n"
          f"  {OUT_DIR / 'tfidf_top_words_by_beef.csv'}")


if __name__ == "__main__":
    main()
