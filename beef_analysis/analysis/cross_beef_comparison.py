"""
cross_beef_comparison.py

各analysisスクリプトの出力を突き合わせてビーフ間比較を行う。

1. beef1・2・3（メイン）の対立語彙率・エンゲージメント指標を横並びで比較
2. Drake関与（beef1・2） vs 非Drake（beef3）でパターンが一致するかを確認
3. 補助分析（beef4・5）で歴史的ビーフの対立言及が現在まで持続しているかを確認
4. 頑健性チェック：PRE/PEAK/POSTの区切りを±14日ずらしても
   対立語彙率・相手言及率の傾向（PRE<PEAK, PEAK>POST等）が変わらないかを確認

前提: conflict_vocab.py / engagement_metrics.py / viewcount_analysis.py を
先に実行し、data/processed/ 配下の各種summary CSVが揃っていること。

出力:
  data/processed/cross_beef_period_summary.csv
  data/processed/cross_beef_drake_bias.csv
  data/processed/cross_beef_supplement_persistence.csv
  data/processed/sensitivity_shift_check.csv

使い方:
  python cross_beef_comparison.py
"""

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conflict_vocab import has_conflict_vocab, has_opponent_mention  # noqa: E402

BEEF_ROOT = Path(__file__).resolve().parents[1]
BEEFS_CSV = BEEF_ROOT / "config" / "beefs.csv"
DATA_PROCESSED = BEEF_ROOT / "data" / "processed"

DRAKE_BEEFS = ["beef1", "beef2"]
NON_DRAKE_BEEFS = ["beef3"]
SHIFT_DAYS = 14


def load_beefs():
    with open(BEEFS_CSV, newline="", encoding="utf-8") as f:
        return {r["beef_id"]: r for r in csv.DictReader(f)}


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def compute_periods(start_date, decisive_date, shift_days=0):
    shift = timedelta(days=shift_days)
    return {
        "PRE": (start_date - timedelta(days=90) + shift, start_date - timedelta(seconds=1) + shift),
        "PEAK": (start_date + shift, decisive_date + timedelta(days=30) + shift),
        "POST": (decisive_date + timedelta(days=31) + shift, decisive_date + timedelta(days=120) + shift),
    }


def classify(published_at, periods):
    for label, (start, end) in periods.items():
        if start <= published_at <= end:
            return label
    return "NA"


def build_period_summary():
    conflict_path = DATA_PROCESSED / "conflict_vocab_summary.csv"
    engagement_path = DATA_PROCESSED / "engagement_by_period.csv"
    if not conflict_path.exists() or not engagement_path.exists():
        print("警告: conflict_vocab_summary.csv / engagement_by_period.csv が見つかりません。"
              "先に conflict_vocab.py / engagement_metrics.py を実行してください。")
        return None

    # period列には補助beefの"NA"という文字列値が入るため、欠損値として解釈されないよう
    # keep_default_na=Falseにし、空文字のみを欠損として扱う
    conflict = pd.read_csv(conflict_path, keep_default_na=False, na_values=[""])
    engagement = pd.read_csv(engagement_path, keep_default_na=False, na_values=[""])
    merged = pd.merge(conflict, engagement, on=["beef_id", "period", "n_comments"], how="outer")

    out_path = DATA_PROCESSED / "cross_beef_period_summary.csv"
    merged.to_csv(out_path, index=False)
    print(f"ビーフ×期間の統合サマリー -> {out_path}")
    return merged


def drake_bias_check(merged):
    if merged is None:
        return
    main = merged[merged["beef_id"].isin(DRAKE_BEEFS + NON_DRAKE_BEEFS)].copy()
    main["group"] = main["beef_id"].apply(lambda b: "drake_involved" if b in DRAKE_BEEFS else "non_drake")
    period_order = pd.Categorical(main["period"], categories=["PRE", "PEAK", "POST"], ordered=True)
    main["period"] = period_order
    main = main.sort_values(["group", "beef_id", "period"])

    out_path = DATA_PROCESSED / "cross_beef_drake_bias.csv"
    main.to_csv(out_path, index=False)
    print(f"\nDrake関与 vs 非Drake 比較 -> {out_path}")
    print(main[["beef_id", "group", "period", "n_comments", "pct_conflict_vocab", "pct_opponent_mention"]].to_string(index=False))


def supplement_persistence_check(merged):
    if merged is None:
        return
    supplement = merged[merged["beef_id"].isin(["beef4", "beef5"])]
    out_path = DATA_PROCESSED / "cross_beef_supplement_persistence.csv"
    supplement.to_csv(out_path, index=False)
    print(f"\n補助分析（歴史的ビーフの現在の言説）-> {out_path}")
    print(supplement[["beef_id", "period", "n_comments", "pct_conflict_vocab", "pct_opponent_mention"]].to_string(index=False))


def sensitivity_check():
    beefs = load_beefs()
    rows = []

    for beef_id in ["beef1", "beef2", "beef3"]:
        beef_row = beefs[beef_id]
        in_path = DATA_PROCESSED / f"{beef_row['folder_name']}_comments_clean.csv"
        if not in_path.exists():
            print(f"[{beef_id}] {in_path} が見つからないため感度分析をスキップ")
            continue

        df = pd.read_csv(in_path)
        if df.empty:
            continue
        df["published_at_dt"] = pd.to_datetime(df["published_at"], utc=True)
        df["has_conflict_vocab"] = df["text_clean"].fillna("").apply(has_conflict_vocab)
        df["has_opponent_mention"] = df.apply(
            lambda r: has_opponent_mention(str(r["text_clean"]), beef_id, r["artist"]), axis=1
        )

        start_date = parse_date(beef_row["start_date"])
        decisive_date = parse_date(beef_row["decisive_date"])

        for shift_days in [-SHIFT_DAYS, 0, SHIFT_DAYS]:
            periods = compute_periods(start_date, decisive_date, shift_days)
            shifted_period = df["published_at_dt"].apply(lambda d: classify(d, periods))
            for period in ["PRE", "PEAK", "POST"]:
                sub = df[shifted_period == period]
                n = len(sub)
                rows.append({
                    "beef_id": beef_id,
                    "shift_days": shift_days,
                    "period": period,
                    "n_comments": n,
                    "pct_conflict_vocab": round(sub["has_conflict_vocab"].mean() * 100, 2) if n else None,
                    "pct_opponent_mention": round(sub["has_opponent_mention"].mean() * 100, 2) if n else None,
                })

    if not rows:
        print("感度分析: 対象データが0件でした。")
        return

    out_path = DATA_PROCESSED / "sensitivity_shift_check.csv"
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"\n頑健性チェック（PRE/PEAK/POSTの区切りを±{SHIFT_DAYS}日ずらした場合）-> {out_path}")
    print(result.to_string(index=False))


def main():
    merged = build_period_summary()
    drake_bias_check(merged)
    supplement_persistence_check(merged)
    sensitivity_check()


if __name__ == "__main__":
    main()
