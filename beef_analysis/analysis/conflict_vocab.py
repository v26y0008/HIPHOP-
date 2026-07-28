"""
conflict_vocab.py

data/processed/{beef_folder}_comments_clean.csv（preprocess.py の出力）を使い、
対立語彙・相手アーティスト名の言及率を期間（PRE/PEAK/POST）ごとに計算し、
カイ二乗検定でPRE->PEAK->POSTの変化を検証する。

グラフ描画自体は visualize/plot_timeseries.py が
本スクリプトの出力CSVを読んで行う。

出力:
  data/processed/conflict_vocab_summary.csv   期間別の出現率
  data/processed/conflict_vocab_chisq.csv     カイ二乗検定結果

使い方:
  python conflict_vocab.py
"""

import csv
import re
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

BEEF_ROOT = Path(__file__).resolve().parents[1]
BEEFS_CSV = BEEF_ROOT / "config" / "beefs.csv"
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"

CONFLICT_VOCAB = [
    "diss", "beef", "dissed", "destroyed", "bodied", "ate",
    "cooked", "buried", "clowned", "exposed", "ended",
    "won", "lost", "winner", "loser", "battle", "respond",
    "response", "shot", "fired", "bars", "ratio",
]

# 各beefの「side」キー -> そのsideの動画コメントで相手を指す語彙
OPPONENT_NAMES = {
    "beef1": {
        "kendrick_side": ["drake", "aubrey", "champagne papi", "drizzy"],
        "drake_side":    ["kendrick", "kdot", "k dot", "pglang", "compton"],
    },
    "beef2": {
        "pusha_side": ["drake", "aubrey", "drizzy"],
        "drake_side": ["pusha", "pusha t", "king push", "clipse"],
    },
    "beef3": {
        "wayne_side":   ["birdman", "baby", "cash money", "cm"],
        "birdman_side": ["wayne", "weezy", "tunechi", "young money"],
    },
    "beef4": {
        "jayz_side": ["nas", "nasir", "escobar"],
        "nas_side":  ["jay", "jay-z", "jigga", "hov"],
    },
    "beef5": {
        "2pac_side":  ["biggie", "notorious", "b.i.g", "smalls"],
        "biggie_side": ["pac", "2pac", "tupac", "shakur"],
    },
}

# 動画のartist名 -> 上記OPPONENT_NAMESのどのsideキーを使うか
ARTIST_TO_SIDE = {
    "beef1": {"Kendrick Lamar": "kendrick_side", "Drake": "drake_side"},
    "beef2": {"Pusha T": "pusha_side", "Drake": "drake_side"},
    "beef3": {"Lil Wayne": "wayne_side", "Birdman": "birdman_side"},
    "beef4": {"Jay-Z": "jayz_side", "Nas": "nas_side"},
    "beef5": {"2Pac": "2pac_side", "Biggie": "biggie_side"},
}


def compile_vocab_pattern(terms):
    escaped = [re.escape(t) for t in terms]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


CONFLICT_PATTERN = compile_vocab_pattern(CONFLICT_VOCAB)


def has_conflict_vocab(text):
    return bool(CONFLICT_PATTERN.search(text))


def has_opponent_mention(text, beef_id, artist):
    side_key = ARTIST_TO_SIDE.get(beef_id, {}).get(artist)
    if side_key is None:
        return False
    terms = OPPONENT_NAMES[beef_id][side_key]
    pattern = compile_vocab_pattern(terms)
    return bool(pattern.search(text))


def load_beefs():
    with open(BEEFS_CSV, newline="", encoding="utf-8") as f:
        return {r["beef_id"]: r for r in csv.DictReader(f)}


def analyze_beef(beef_id, beef_row):
    in_path = DATA_PROCESSED / f"{beef_row['folder_name']}_comments_clean.csv"
    if not in_path.exists():
        print(f"[{beef_id}] 前処理済みファイルが見つかりません: {in_path}（スキップ。先に preprocess.py を実行してください）")
        return None, None

    # period列には補助beefで"NA"という文字列値が入るため、欠損値として解釈されないよう
    # keep_default_na=Falseにし、空文字のみを欠損として扱う
    df = pd.read_csv(in_path, keep_default_na=False, na_values=[""])
    if df.empty:
        print(f"[{beef_id}] コメントが0件（スキップ）")
        return None, None

    df["has_conflict_vocab"] = df["text_clean"].fillna("").apply(has_conflict_vocab)
    df["has_opponent_mention"] = df.apply(
        lambda r: has_opponent_mention(str(r["text_clean"]), beef_id, r["artist"]), axis=1
    )

    summary_rows = []
    periods = ["PRE", "PEAK", "POST"] if beef_row["beef_role"] == "main" else sorted(df["period"].unique())
    for period in periods:
        sub = df[df["period"] == period]
        n = len(sub)
        summary_rows.append({
            "beef_id": beef_id,
            "period": period,
            "n_comments": n,
            "n_conflict_vocab": int(sub["has_conflict_vocab"].sum()) if n else 0,
            "pct_conflict_vocab": round(sub["has_conflict_vocab"].mean() * 100, 2) if n else None,
            "n_opponent_mention": int(sub["has_opponent_mention"].sum()) if n else 0,
            "pct_opponent_mention": round(sub["has_opponent_mention"].mean() * 100, 2) if n else None,
        })

    chisq_rows = []
    if beef_row["beef_role"] == "main":
        for col, label in [("has_conflict_vocab", "conflict_vocab"), ("has_opponent_mention", "opponent_mention")]:
            table = pd.crosstab(df["period"], df[col])
            table = table.reindex(["PRE", "PEAK", "POST"]).dropna(how="all")
            if table.shape[0] < 2 or table.shape[1] < 2:
                print(f"[{beef_id}] {label}: 期間数またはカテゴリ数が不足のためカイ二乗検定をスキップ")
                continue
            chi2, p, dof, _ = chi2_contingency(table)
            chisq_rows.append({
                "beef_id": beef_id, "test": label, "chi2": round(chi2, 4), "dof": dof, "p_value": p,
            })
            print(f"[{beef_id}] {label}: chi2={chi2:.4f}, dof={dof}, p={p:.4g}")

    return summary_rows, chisq_rows


def main():
    beefs = load_beefs()
    all_summary, all_chisq = [], []

    for beef_id, beef_row in beefs.items():
        summary_rows, chisq_rows = analyze_beef(beef_id, beef_row)
        if summary_rows:
            all_summary.extend(summary_rows)
        if chisq_rows:
            all_chisq.extend(chisq_rows)

    if not all_summary:
        print("集計対象が0件でした。")
        return

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_PROCESSED / "conflict_vocab_summary.csv"
    pd.DataFrame(all_summary).to_csv(summary_path, index=False)
    print(f"\n期間別出現率 -> {summary_path}")

    if all_chisq:
        chisq_path = DATA_PROCESSED / "conflict_vocab_chisq.csv"
        pd.DataFrame(all_chisq).to_csv(chisq_path, index=False)
        print(f"カイ二乗検定結果 -> {chisq_path}")


if __name__ == "__main__":
    main()
