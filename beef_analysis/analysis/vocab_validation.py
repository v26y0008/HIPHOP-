"""
vocab_validation.py

conflict_vocab.py の CONFLICT_VOCAB リストが、実際のコメントデータの
高頻度語彙をどれだけカバーしているかを定量的に検証する。

Step 1: 全コメント（beef1/2/3/5、前処理済み）からTF-IDFで上位200語を抽出
Step 2: 上位200語を「対立・勝敗系」「人物言及系」「感情系」「その他」に
        キーワードヒューリスティックで自動分類し、vocab_labels.csv に保存
        （注意：人手によるコーディングではなく、キーワード辞書による自動分類。
        学会発表で使う場合は目視で妥当性を確認すること）
Step 3: 現在のCONFLICT_VOCABが上位200語に何語含まれるかを集計
Step 4: 上位200語のうち「対立・勝敗系」に分類されたがCONFLICT_VOCABに
        含まれていない語（拡張候補）を列挙
Step 5: CONFLICT_VOCAB各語の実際の出現コメント数を数える（上位200語に
        入らなくても実際には使われている語と、ほぼ出現しない「死に語」を区別するため）

出力:
  outputs/stats/vocab_labels.csv
  outputs/stats/vocab_validation.txt

使い方:
  python vocab_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BEEF_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"
OUTPUT_DIR = BEEF_ROOT / "outputs" / "stats"

FOLDERS = [
    "beef1_kendrick_drake",
    "beef2_pusha_drake",
    "beef3_wayne_birdman",
    "beef5_2pac_biggie",
]

TOP_N = 200

# conflict_vocab.py の現行リスト（重複判定用にそのまま複製せず import する）
import csv
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conflict_vocab import CONFLICT_VOCAB, compile_vocab_pattern, has_conflict_vocab  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

BEEFS_CSV = BEEF_ROOT / "config" / "beefs.csv"

DEAD_WORD_THRESHOLD = 5  # 出現コメント数がこれ未満なら「死に語」候補として警告

CONFLICT_KEYWORDS = set(CONFLICT_VOCAB) | {
    "clapped", "murdered", "smoked", "ether", "ethered", "folded", "embarrassed",
    "humiliated", "demolished", "wrecked", "finished", "defeated", "victory",
    "defeat", "versus", "knockout", "ko", "round", "war", "rip",
}
PERSON_KEYWORDS = {
    "drake", "kendrick", "pusha", "wayne", "birdman", "lil", "jay", "nas",
    "biggie", "pac", "tupac", "aubrey", "weezy", "hov", "thug", "kdot",
    "bro", "guy", "dude", "man", "girl", "boy", "him", "he", "his",
}
EMOTION_KEYWORDS = {
    "love", "hate", "crazy", "insane", "sad", "happy", "omg", "lol", "lmao",
    "damn", "wow", "amazing", "best", "worst", "favorite", "fire", "hard",
    "dope", "goat", "legend", "legendary", "iconic", "cold", "sick", "good",
    "great", "beautiful", "sad",
}


def classify_term(term):
    if term in CONFLICT_KEYWORDS:
        return "対立・勝敗系"
    if term in PERSON_KEYWORDS:
        return "人物言及系"
    if term in EMOTION_KEYWORDS:
        return "感情系"
    return "その他"


def load_all_text():
    texts = []
    for folder in FOLDERS:
        path = DATA_PROCESSED / f"{folder}_comments_clean.csv"
        if not path.exists():
            print(f"警告: {path} が見つかりません（スキップ）")
            continue
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        texts.append(df["text_clean"].dropna())
    if not texts:
        return pd.Series(dtype=str)
    return pd.concat(texts, ignore_index=True)


MAIN_BEEFS = ["beef1", "beef2", "beef3"]
BEEF_LABELS = {
    "beef1": "Beef1: Kendrick vs Drake",
    "beef2": "Beef2: Pusha T vs Drake",
    "beef3": "Beef3: Wayne vs Birdman",
}


def dead_word_robustness_check(dead_words):
    """死に語を除外したCONFLICT_VOCABで検定をやり直し、結論が変わるか確認する。"""
    if not dead_words:
        return ["", "死に語が0件のため、除外による頑健性チェックはスキップ。"]

    with open(BEEFS_CSV, newline="", encoding="utf-8") as f:
        beefs = {r["beef_id"]: r for r in csv.DictReader(f)}

    reduced_vocab = [w for w in CONFLICT_VOCAB if w not in dead_words]
    reduced_pattern = compile_vocab_pattern(reduced_vocab)

    lines = [
        "",
        f"=== 死に語（{', '.join(dead_words)}）を除外した場合の頑健性チェック ===",
        f"CONFLICT_VOCAB {len(CONFLICT_VOCAB)}語 -> 除外後 {len(reduced_vocab)}語 で対立語彙判定・カイ二乗検定を再実行",
        "",
    ]

    for beef_id in MAIN_BEEFS:
        folder = beefs[beef_id]["folder_name"]
        path = DATA_PROCESSED / f"{folder}_comments_clean.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df = df[df["period"].isin(["PRE", "PEAK", "POST"])].copy()
        if df.empty:
            continue

        df["has_full"] = df["text_clean"].fillna("").apply(has_conflict_vocab)
        df["has_reduced"] = df["text_clean"].fillna("").apply(lambda t: bool(reduced_pattern.search(t)))

        row = {"beef_id": beef_id}
        for label, col in [("full", "has_full"), ("dead_excluded", "has_reduced")]:
            n = df.groupby("period").size().reindex(["PRE", "PEAK", "POST"]).fillna(0)
            pos = df.groupby("period")[col].sum().reindex(["PRE", "PEAK", "POST"]).fillna(0)
            neg = n - pos
            table = [pos.tolist(), neg.tolist()]
            chi2, p, dof, _ = chi2_contingency(table)
            row[label] = (chi2, p)

        full_chi2, full_p = row["full"]
        red_chi2, red_p = row["dead_excluded"]
        flip = (full_p < 0.05) != (red_p < 0.05)
        status = "!! 結論が変化する !!" if flip else "結論は変化しない"
        lines.append(
            f"{BEEF_LABELS[beef_id]}: 全22語 chi2={full_chi2:.4f} p={full_p:.4g}  |  "
            f"死に語除外(19語) chi2={red_chi2:.4f} p={red_p:.4g}  -> {status}"
        )

    return lines


def main():
    all_text = load_all_text()
    if all_text.empty:
        print("コメントデータが見つかりません。先に preprocess.py を実行してください。")
        return

    vectorizer = TfidfVectorizer(max_features=TOP_N, min_df=5, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_text)
    terms = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

    top200 = pd.DataFrame({"term": terms, "mean_tfidf": mean_scores})
    top200["category"] = top200["term"].apply(classify_term)
    top200 = top200.sort_values("mean_tfidf", ascending=False).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT_DIR / "vocab_labels.csv"
    top200.to_csv(labels_path, index=False)

    covered = [w for w in CONFLICT_VOCAB if w in set(top200["term"])]
    coverage_pct = len(covered) / len(CONFLICT_VOCAB) * 100

    expansion_candidates = top200[
        (top200["category"] == "対立・勝敗系") & (~top200["term"].isin(CONFLICT_VOCAB))
    ]["term"].tolist()

    # 上位200語入りとは無関係に、CONFLICT_VOCAB各語の実際の出現コメント数を数える
    n_docs = len(all_text)
    raw_counts = {}
    for word in CONFLICT_VOCAB:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b")
        raw_counts[word] = int(all_text.fillna("").str.contains(pattern).sum())
    dead_words = sorted(w for w, c in raw_counts.items() if c < DEAD_WORD_THRESHOLD)
    counts_df = pd.DataFrame(
        {"term": list(raw_counts.keys()), "n_comments_containing": list(raw_counts.values())}
    ).sort_values("n_comments_containing", ascending=False)
    counts_path = OUTPUT_DIR / "vocab_raw_occurrence_counts.csv"
    counts_df.to_csv(counts_path, index=False)

    lines = [
        "=== CONFLICT_VOCAB 網羅率の検証 ===",
        "（自動分類：キーワードヒューリスティックによる。人手コーディングではないため目視確認を推奨）",
        "",
        f"CONFLICT_VOCAB {len(CONFLICT_VOCAB)}語のうち、コメント上位{TOP_N}語彙に含まれる語: "
        f"{len(covered)}語 ({coverage_pct:.0f}%)",
        f"  含まれる語: {', '.join(sorted(covered))}",
        "",
        f"上位{TOP_N}語彙に含まれるがCONFLICT_VOCABに未収録の対立語彙候補（{len(expansion_candidates)}語）:",
        f"  {', '.join(expansion_candidates) if expansion_candidates else '(なし)'}",
        "",
        "カテゴリ別内訳（上位200語中）:",
    ]
    for cat, count in top200["category"].value_counts().items():
        lines.append(f"  {cat}: {count}語")

    lines += [
        "",
        f"=== CONFLICT_VOCAB各語の実出現数（全{n_docs}コメント中） ===",
        "（上位200語に入らなくても実際に一定数使われている語と、ほぼ出現しない「死に語」を区別する）",
        counts_df.to_string(index=False),
        "",
        f"出現コメント数が{DEAD_WORD_THRESHOLD}件未満の「死に語」候補（{len(dead_words)}語）:",
        f"  {', '.join(dead_words) if dead_words else '(なし)'}",
        "  → これらは今回のコーパスではほとんど機能していない語彙。次回のCONFLICT_VOCAB改訂時に",
        "    削除または類語への置き換えを検討する候補。",
    ]

    lines += dead_word_robustness_check(dead_words)

    text_path = OUTPUT_DIR / "vocab_validation.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"語彙ラベル一覧 -> {labels_path}")
    print(f"検証結果 -> {text_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
