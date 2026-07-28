"""
viewcount_analysis.py (v2 追加モジュール)

問い：ビーフは当事者アーティストの「再生される力」を変えるか？

核心指標：再生効率 = view_count / days_since_release
  （リリースからの経過日数で割ることで「1日あたり平均再生数」を近似）
  → 古い曲が単純に再生数が多いというバイアスを除去

【重要】カタログ曲は「全キャリア最大ヒット」ではなく「ビーフ1〜3年前のリリースの中での
再生数上位」を選定している（config/songs_v2.csvのnote参照）。これは、全キャリア最大ヒット
（10年以上かけて再生数を積み上げた曲）とPOST曲（まだ1〜2年の曲）を比較すると、
再生効率という指標で正規化してもなお「経過時間が長いほど有利」という残余バイアスが
残ることが判明したため。各beef_idの最初の単発catalog行（例: b1_catalog_kendrick=HUMBLE.）は
コメント分析（conflict_vocab_analysis.py）用に維持しているが全キャリア最大ヒットであり
recency-matchedではないため、本スクリプトの再生効率比較からは除外する。

分析1：アーティスト別 カタログ曲群 vs POST曲群 の再生効率比較（Mann-Whitney U検定）
分析2：勝者 vs 敗者 で再生効率の変化方向が異なるか
分析3：ビーフ曲の再生効率はカタログ曲より高いか（ビーフ効果の再生数版確認）

使い方:
  python viewcount_analysis.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "all_video_metadata_v2.csv"
SONGS_CSV = BEEF_ROOT / "config" / "songs_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"

BEEF_OUTCOME = {
    "b1": {"winner": "kendrick", "loser": "drake", "verdict": "clear"},
    "b2": {"winner": "pusha", "loser": "drake", "verdict": "clear"},
    "b4": {"winner": "eminem", "loser": "mgk", "verdict": "clear"},
    "b5": {"winner": "wayne", "loser": "birdman", "verdict": "clear", "outcome_type": "legal"},
    "b6": {"winner": "megan", "loser": "nicki", "verdict": "disputed"},
}
BEEF_NAMES = {
    "b1": "Kendrick vs Drake", "b2": "Pusha T vs Drake", "b4": "Eminem vs MGK",
    "b5": "Wayne vs Birdman", "b6": "Megan vs Nicki",
}

# 全キャリア最大ヒット（recency-matchedでない）単発catalog行。コメント分析専用に残しており、
# 本スクリプトの再生効率比較（catalogとpostの経過時間バイアスを避けるため）からは除外する
LEGACY_CATALOG_KEYS = {
    "b1_catalog_kendrick", "b1_catalog_drake", "b2_catalog_pusha", "b2_catalog_drake",
    "b4_catalog_eminem", "b4_catalog_mgk", "b5_catalog_wayne", "b5_catalog_birdman",
    "b6_catalog_megan", "b6_catalog_nicki",
}


def compute_view_efficiency(df):
    df = df.copy()
    df["release_date_dt"] = pd.to_datetime(df["release_date"])
    df["collected_at_dt"] = pd.to_datetime(df["collected_at"].str[:10])
    df["days_since_release"] = (df["collected_at_dt"] - df["release_date_dt"]).dt.days.clip(lower=1)
    df["view_efficiency"] = df["view_count"] / df["days_since_release"]
    df["like_efficiency"] = df["like_count"] / df["days_since_release"]
    return df


def assign_win_loss(df):
    def get_wl(row):
        outcome = BEEF_OUTCOME.get(row["beef_id"], {})
        side = str(row["artist_side"]).lower()
        if side == outcome.get("winner", ""):
            return "winner"
        if side == outcome.get("loser", ""):
            return "loser"
        return "unknown"

    df = df.copy()
    df["win_loss_side"] = df.apply(get_wl, axis=1)
    df["verdict"] = df["beef_id"].map(lambda b: BEEF_OUTCOME.get(b, {}).get("verdict", ""))
    df["outcome_type"] = df["beef_id"].map(lambda b: BEEF_OUTCOME.get(b, {}).get("outcome_type", "rap_battle"))
    return df


def analyze_pre_post_by_artist(df):
    print("\n" + "=" * 60 + "\n=== 分析1：アーティスト別 カタログ vs POST 再生効率 ===\n" + "=" * 60)

    results = []
    for beef_id in sorted(df["beef_id"].unique()):
        outcome = BEEF_OUTCOME.get(beef_id, {})
        for side in [outcome.get("winner", ""), outcome.get("loser", "")]:
            if not side:
                continue
            sub = df[(df["beef_id"] == beef_id) & (df["artist_side"] == side)]
            cat = sub[sub["track_category"] == "catalog"]["view_efficiency"].dropna()
            post = sub[sub["track_category"] == "post"]["view_efficiency"].dropna()

            if len(cat) < 2 or len(post) < 2:
                print(f"\n  {beef_id} {side}: データ不足 (catalog={len(cat)}, post={len(post)})")
                continue

            cat_med, post_med = cat.median(), post.median()
            change_pct = (post_med - cat_med) / cat_med * 100
            stat, p = mannwhitneyu(cat, post, alternative="two-sided")
            wl = "勝者" if side == outcome.get("winner") else "敗者"
            verdict_note = "(disputed)" if outcome.get("verdict") == "disputed" else "(clear)"

            print(f"\n  {beef_id} {side}（{wl}）{verdict_note}: n_catalog={len(cat)}, n_post={len(post)}")
            print(f"    カタログ中央値: {cat_med:,.0f} 再生/日   POST中央値: {post_med:,.0f} 再生/日")
            print(f"    変化率: {change_pct:+.1f}%   p値: {p:.4f}")

            results.append({
                "beef_id": beef_id, "artist_side": side, "win_loss": wl,
                "verdict": outcome.get("verdict", ""), "n_catalog": len(cat), "n_post": len(post),
                "catalog_median_eff": cat_med, "post_median_eff": post_med,
                "change_pct": change_pct, "p_value": p, "significant": p < 0.05,
                "direction": "up" if change_pct > 0 else "down",
            })

    return pd.DataFrame(results)


def analyze_winner_vs_loser_change(results_df):
    print("\n" + "=" * 60 + "\n=== 分析2：勝者 vs 敗者 の再生効率変化方向 ===\n" + "=" * 60)

    for verdict_filter, label in [(None, "全ビーフ"), ("clear", "clear verdictのみ")]:
        sub = results_df if verdict_filter is None else results_df[results_df["verdict"] == verdict_filter]
        winner_changes = sub[sub["win_loss"] == "勝者"]["change_pct"]
        loser_changes = sub[sub["win_loss"] == "敗者"]["change_pct"]

        print(f"\n[{label}]")
        print(f"  勝者の再生効率変化: 平均{winner_changes.mean():+.1f}% (n={len(winner_changes)})")
        print(f"  敗者の再生効率変化: 平均{loser_changes.mean():+.1f}% (n={len(loser_changes)})")
        if len(winner_changes):
            print(f"  勝者で上昇したケース: {(winner_changes > 0).sum()}/{len(winner_changes)}")
        if len(loser_changes):
            print(f"  敗者で下落したケース: {(loser_changes < 0).sum()}/{len(loser_changes)}")

        if len(winner_changes) >= 2 and len(loser_changes) >= 2:
            stat, p = mannwhitneyu(winner_changes, loser_changes, alternative="greater")
            print(f"  Mann-Whitney U（勝者>敗者、片側）: p={p:.4f}")


def analyze_beef_track_boost(df):
    print("\n" + "=" * 60 + "\n=== 分析3：ビーフ曲の再生効率 vs カタログ曲 ===\n" + "=" * 60)

    cat = df[df["track_category"] == "catalog"]["view_efficiency"].dropna()
    beef = df[df["track_category"] == "beef"]["view_efficiency"].dropna()
    post = df[df["track_category"] == "post"]["view_efficiency"].dropna()

    print(f"  カタログ曲: 中央値 {cat.median():,.0f} 再生/日 (n={len(cat)})")
    print(f"  ビーフ曲:   中央値 {beef.median():,.0f} 再生/日 (n={len(beef)})")
    print(f"  POST曲:     中央値 {post.median():,.0f} 再生/日 (n={len(post)})")

    if len(beef) >= 2 and len(cat) >= 2:
        stat1, p1 = mannwhitneyu(beef, cat, alternative="greater")
        print(f"\n  ビーフ曲 > カタログ曲: p={p1:.4f}")
    if len(post) >= 2 and len(cat) >= 2:
        stat2, p2 = mannwhitneyu(post, cat, alternative="two-sided")
        print(f"  POST曲 vs カタログ曲: p={p2:.4f}")


def main():
    if not IN_PATH.exists():
        print(f"エラー: {IN_PATH} が見つかりません。先に fetch_video_stats.py を実行してください。")
        return

    df = pd.read_csv(IN_PATH)
    for col in ["view_count", "like_count", "comment_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # all_video_metadata_v2.csv には artist_side が無いため songs_v2.csv から補完する
    songs = pd.read_csv(SONGS_CSV, keep_default_na=False, na_values=[""])[["key", "artist_side"]]
    df = df.merge(songs, on="key", how="left")

    n_before = len(df)
    df = df[~df["key"].isin(LEGACY_CATALOG_KEYS)]
    print(f"全キャリア最大ヒットのcatalog行を除外: {n_before} -> {len(df)}件"
          f"（除外{n_before - len(df)}件、詳細はLEGACY_CATALOG_KEYS参照）")

    df = compute_view_efficiency(df)
    df = assign_win_loss(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "video_stats_with_efficiency_v2.csv", index=False)

    results_df = analyze_pre_post_by_artist(df)
    analyze_winner_vs_loser_change(results_df)
    analyze_beef_track_boost(df)

    results_df.to_csv(OUT_DIR / "viewcount_results_v2.csv", index=False)
    print(f"\n結果を保存しました -> {OUT_DIR / 'viewcount_results_v2.csv'}, "
          f"{OUT_DIR / 'video_stats_with_efficiency_v2.csv'}")


if __name__ == "__main__":
    main()
