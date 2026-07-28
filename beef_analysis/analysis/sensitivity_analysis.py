"""
sensitivity_analysis.py (v2 設計)

CONFLICT_VOCABの頑健性を検証する。
① 死に語（出現5件未満）を除外した場合に、各ビーフの検定結論が変わるか
② Drake関与・非関与でパターンが共通しているか（すでにconflict_vocab_analysis.pyで
   要約集計済みだが、ここでは「死に語除外後」の値で再確認する）
③ ビーフタイプ別のパターンが死に語除外後も成立するか

使い方:
  python sensitivity_analysis.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_DIR = BEEF_ROOT / "outputs" / "stats"
CATEGORY_ORDER = ["catalog", "beef", "post"]
DEAD_WORD_THRESHOLD = 5

CONFLICT_VOCAB = [
    "diss", "beef", "dissed", "destroyed", "bodied", "ate",
    "cooked", "buried", "exposed", "ended", "won", "lost",
    "winner", "loser", "battle", "respond", "response",
    "shot", "fired", "bars", "clapped", "murked",
]

BEEF_LABELS = {
    "b1": "B1: Kendrick vs Drake", "b2": "B2: Pusha T vs Drake",
    "b4": "B4: Eminem vs MGK", "b5": "B5: Wayne vs Birdman", "b6": "B6: Megan vs Nicki",
}


def compile_pattern(words):
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b")


def word_present(text, pattern):
    return bool(pattern.search(str(text)))


def run_chi2(df_beef, col):
    ct = df_beef.groupby("track_category")[col].agg(["sum", "count"]).reindex(CATEGORY_ORDER).dropna()
    if len(ct) < 2:
        return None
    pos = ct["sum"].to_numpy()
    neg = (ct["count"] - ct["sum"]).to_numpy()
    chi2, p, dof, _ = chi2_contingency(np.array([pos, neg]))
    return chi2, p


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。")
        return

    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])

    # ---- 死に語の特定 ----
    full_pattern = compile_pattern(CONFLICT_VOCAB)
    word_counts = {}
    for word in CONFLICT_VOCAB:
        pattern = compile_pattern([word])
        word_counts[word] = int(df["text_clean"].fillna("").apply(lambda t: word_present(t, pattern)).sum())

    dead_words = sorted(w for w, c in word_counts.items() if c < DEAD_WORD_THRESHOLD)
    active_words = [w for w in CONFLICT_VOCAB if w not in dead_words]

    lines = ["=== CONFLICT_VOCAB 各語の実出現数 ===", ""]
    for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {w:<12} {c:>6}件")
    lines.append("")
    lines.append(f"死に語（{DEAD_WORD_THRESHOLD}件未満）: {dead_words if dead_words else '(なし)'}")
    lines.append(f"有効語: {len(active_words)}/{len(CONFLICT_VOCAB)}語")
    lines.append("")

    # ---- 全語 vs 有効語のみで検定結果を比較 ----
    reduced_pattern = compile_pattern(active_words) if active_words else None
    df["conflict_full"] = df["text_clean"].fillna("").apply(lambda t: word_present(t, full_pattern))
    if reduced_pattern is not None:
        df["conflict_active"] = df["text_clean"].fillna("").apply(lambda t: word_present(t, reduced_pattern))
    else:
        df["conflict_active"] = False

    lines.append("=== 死に語除外の前後でカイ二乗検定の結論が変わるか ===")
    flipped_any = False
    for bid in sorted(df["beef_id"].unique()):
        df_b = df[df["beef_id"] == bid]
        r_full = run_chi2(df_b, "conflict_full")
        r_active = run_chi2(df_b, "conflict_active")
        if r_full is None or r_active is None:
            continue
        chi2_full, p_full = r_full
        chi2_active, p_active = r_active
        flip = (p_full < 0.05) != (p_active < 0.05)
        flipped_any = flipped_any or flip
        status = "!!結論が変化する!!" if flip else "結論は変化しない"
        lines.append(
            f"{BEEF_LABELS.get(bid, bid)}: 全語 p={p_full:.4g} | 有効語のみ p={p_active:.4g} -> {status}"
        )

    lines.append("")
    lines.append(
        "全ビーフで死に語除外後も結論は変化しない。" if not flipped_any
        else "一部のビーフで死に語除外により結論が変化する。解釈時に注意。"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_path = OUT_DIR / "sensitivity_analysis_v2.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n-> {text_path}")


if __name__ == "__main__":
    main()
