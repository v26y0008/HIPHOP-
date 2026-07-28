"""
engagement_analysis.py (v2 設計)

再生数を使った熱量分析。
viewCountは取得時点のスナップショットであり「古い曲ほど多い」バイアスがあるため、
days_since_release（コメント投稿日 - 動画公開日の中央値）を共変量とした
偏回帰分析で、経過日数を統制してもビーフ曲の熱量（comment_per_view）が
高いままかどうかを検証する。

使い方:
  python engagement_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BEEF_ROOT = Path(__file__).resolve().parents[1]
COMMENTS_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
VIDEO_STATS_PATH = BEEF_ROOT / "data" / "processed" / "all_video_metadata_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"


def main():
    if not COMMENTS_PATH.exists() or not VIDEO_STATS_PATH.exists():
        print(f"エラー: {COMMENTS_PATH} または {VIDEO_STATS_PATH} が見つかりません。"
              "先に preprocess.py / fetch_video_stats.py を実行してください。")
        return

    comments = pd.read_csv(COMMENTS_PATH, keep_default_na=False, na_values=[""])
    video_stats = pd.read_csv(VIDEO_STATS_PATH)

    video_agg = comments.groupby("video_id").agg(
        n_comments_collected=("comment_id", "count"),
        avg_length=("text_clean", lambda x: x.str.len().mean()),
        reply_rate=("reply_count", "mean"),
        like_rate=("like_count", "mean"),
        beef_id=("beef_id", "first"),
        beef_type=("beef_type", "first"),
        track_category=("track_category", "first"),
        drake_involved=("drake_involved", "first"),
        days_since_release=("days_since_release", "median"),
        artist_name=("artist_name", "first"),
        song_name=("song_name", "first"),
    ).reset_index()

    df = video_agg.merge(video_stats[["video_id", "view_count", "like_count", "comment_count"]], on="video_id")
    df["comment_per_view"] = df["n_comments_collected"] / df["view_count"].clip(lower=1)
    df["like_per_view"] = df["like_count"] / df["view_count"].clip(lower=1)

    # ============================================================
    # 分析A: カテゴリ別 comment_per_view
    # ============================================================
    print("=== カテゴリ別 comment_per_view ===")
    print(df.groupby("track_category")["comment_per_view"].describe())

    # ============================================================
    # 分析B: days_since_release を共変量とした偏回帰
    # 「経過日数を統制した上でもビーフ曲はcomment_per_viewが高いか」
    # ============================================================
    df["is_beef_track"] = (df["track_category"] == "beef").astype(int)
    df["log_days"] = np.log1p(df["days_since_release"].clip(lower=1))
    df["log_cpv"] = np.log1p(df["comment_per_view"] * 1e5)

    X = df[["is_beef_track", "log_days"]].to_numpy()
    y = df["log_cpv"].to_numpy()
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)

    print("\n偏回帰結果（log_cpv ~ is_beef_track + log_days）:")
    print(f"  is_beef_track 係数: {reg.coef_[0]:.4f}")
    print(f"  log_days 係数: {reg.coef_[1]:.4f}")
    print(f"  切片: {reg.intercept_:.4f}")
    print(f"  R^2: {r2:.4f}  (n={len(df)})")
    print("  -> is_beef_track の係数が正なら、経過日数を統制してもビーフ曲の熱量が高い")

    # 簡易的なp値算出（OLSのt検定、statsmodelsが無い環境向けに手計算）
    n, k = X.shape
    X_design = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    residuals = y - X_design @ beta
    dof = n - k - 1
    mse = np.sum(residuals ** 2) / dof
    cov_beta = mse * np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se
    from scipy import stats as scipy_stats
    p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), dof))
    print(f"  is_beef_track: t={t_stats[1]:.3f}, p={p_values[1]:.4g}")
    print(f"  log_days: t={t_stats[2]:.3f}, p={p_values[2]:.4g}")

    # ============================================================
    # 分析C: ビーフ別 カタログ vs ビーフ曲 comment_per_view 比較
    # ============================================================
    print("\n=== ビーフ別 カタログ vs ビーフ曲 comment_per_view ===")
    comparison_rows = []
    for bid in sorted(df["beef_id"].unique()):
        cat_cpv = df[(df["beef_id"] == bid) & (df["track_category"] == "catalog")]["comment_per_view"].mean()
        beef_cpv = df[(df["beef_id"] == bid) & (df["track_category"] == "beef")]["comment_per_view"].mean()
        post_cpv = df[(df["beef_id"] == bid) & (df["track_category"] == "post")]["comment_per_view"].mean()
        ratio = beef_cpv / cat_cpv if pd.notna(cat_cpv) and cat_cpv > 0 else float("nan")
        print(f"  {bid}: catalog={cat_cpv:.6f}, beef={beef_cpv:.6f}, post={post_cpv:.6f}, beef/catalog比={ratio:.1f}x")
        comparison_rows.append({
            "beef_id": bid, "catalog_cpv": cat_cpv, "beef_cpv": beef_cpv,
            "post_cpv": post_cpv, "beef_to_catalog_ratio": ratio,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "video_level_analysis_v2.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(OUT_DIR / "cpv_comparison_by_beef_v2.csv", index=False)
    print(f"\n動画別データ -> {OUT_DIR / 'video_level_analysis_v2.csv'}")
    print(f"ビーフ別比較 -> {OUT_DIR / 'cpv_comparison_by_beef_v2.csv'}")


if __name__ == "__main__":
    main()
