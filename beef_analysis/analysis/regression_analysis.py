"""
regression_analysis.py (v2 追加モジュール)

目的変数：対立語彙率 pct_conflict（動画単位の集計値、ビーフ曲のみ）
説明変数：ビーフタイプ・勝敗・Drake関与・返信率・いいね率・平均コメント長・再生効率

問い：何がビーフ曲コメントの対立度を説明するか？
      →「ビーフタイプ」と「勝敗」どちらが効くか定量化する

重要な注意（実行前から分かっている構造的制約）:
  ビーフ曲カテゴリで実コメントが取得できた動画はn=7しかない
  （B2のPusha T側 Story of Adidon はコメント欄無効のため欠損、
   B5のBirdman側は明確なdiss曲が存在しないため元々データなし）。
  7特徴量+定数項=8パラメータをn=7で推定するのは自由度が不足しており、
  OLSは完全/準完全な過学習（rank-deficient、SEが発散/NaN）になりうる。
  そのため本スクリプトは
    (1) 指示書通りのフル7特徴量モデル
    (2) 自由度を確保した縮小2特徴量モデル（is_rap_battle, is_winnerのみ）
  の両方を実行し、後者を主たる解釈対象とする。

使い方:
  python regression_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Meiryo"
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

BEEF_ROOT = Path(__file__).resolve().parents[1]
VIDEO_LEVEL = BEEF_ROOT / "data" / "processed" / "video_level_analysis_v2.csv"
WIN_LOSS_COMMENTS = BEEF_ROOT / "data" / "processed" / "comments_with_win_loss_v2.csv"
EFFICIENCY = BEEF_ROOT / "data" / "processed" / "video_stats_with_efficiency_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

FEATURES_FULL = [
    "is_rap_battle", "is_winner", "is_drake",
    "reply_rate", "like_rate", "avg_length", "view_efficiency",
]
FEATURES_REDUCED = ["is_rap_battle", "is_winner"]
TARGET = "pct_conflict"

FEAT_LABELS = {
    "is_rap_battle": "ラップバトル型",
    "is_winner": "勝者側",
    "is_drake": "Drake関与",
    "reply_rate": "返信率",
    "like_rate": "いいね率",
    "avg_length": "平均コメント長",
    "view_efficiency": "再生効率",
}


def build_dataset():
    vl = pd.read_csv(VIDEO_LEVEL, keep_default_na=False, na_values=[""])
    beef_df = vl[vl["track_category"] == "beef"].copy()

    # pct_conflict と win_loss_side は comments_with_win_loss_v2.csv から動画単位で集計
    cw = pd.read_csv(WIN_LOSS_COMMENTS, keep_default_na=False, na_values=[""])
    cw_beef = cw[cw["track_category"] == "beef"]
    agg = cw_beef.groupby("video_id").agg(
        pct_conflict=("has_conflict", "mean"),
        win_loss_side=("win_loss_side", "first"),
    ).reset_index()
    agg["pct_conflict"] = agg["pct_conflict"] * 100

    beef_df = beef_df.merge(agg, on="video_id", how="left")

    # view_efficiency は video_stats_with_efficiency_v2.csv から
    eff = pd.read_csv(EFFICIENCY, keep_default_na=False, na_values=[""])
    eff_beef = eff[eff["track_category"] == "beef"][["video_id", "view_efficiency"]]
    beef_df = beef_df.merge(eff_beef, on="video_id", how="left")

    beef_df["is_rap_battle"] = (beef_df["beef_type"] == "rap_battle").astype(int)
    beef_df["is_winner"] = (beef_df["win_loss_side"] == "winner").astype(int)
    beef_df["is_drake"] = beef_df["drake_involved"].astype(str).str.lower().isin(["true", "1"]).astype(int)

    return beef_df


def run_ols(model_df, features, label):
    X = model_df[features].values.astype(float)
    y = model_df[TARGET].values.astype(float)
    n, k = X.shape

    print(f"\n{'=' * 60}\n{label} (n={n}, 特徴量数={k}, 自由度={n - k - 1})\n{'=' * 60}")
    if n - k - 1 < 1:
        print(f"[警告] 自由度が{n - k - 1}以下で残差の分散が推定できません。"
              f"このモデルの p値/R²は解釈不能です（過学習・rank-deficient）。")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_const = sm.add_constant(X_scaled)

    ols = sm.OLS(y, X_const).fit()
    print(ols.summary())

    coef_df = pd.DataFrame({
        "feature": ["const"] + features,
        "coef": ols.params,
        "p_value": ols.pvalues,
        "significant": ols.pvalues < 0.05,
    }).sort_values("coef", key=abs, ascending=False)
    print(f"\n=== 係数（標準化済み、絶対値順）: {label} ===")
    print(coef_df.to_string())

    # LOO-CV
    if n > k + 1:
        loo = LeaveOneOut()
        y_pred_loo = []
        for train_idx, test_idx in loo.split(X_scaled):
            reg = LinearRegression()
            reg.fit(X_scaled[train_idx], y[train_idx])
            y_pred_loo.append(reg.predict(X_scaled[test_idx])[0])
        r2_loo = r2_score(y, y_pred_loo)
    else:
        r2_loo = float("nan")
        print("[注記] n <= 特徴量数+1 のためLOO-CVは実行不能（訓練データ不足）")

    print(f"\nLOO-CV R²: {r2_loo:.3f}" if not np.isnan(r2_loo) else "\nLOO-CV R²: 計算不能")
    print(f"OLS R²: {ols.rsquared:.3f}（n={n}が小さいため過大評価されやすい点に注意）")

    return ols, coef_df, r2_loo


def plot_coefficients(coef_df, ols, r2_loo, out_path, title):
    plot_df = coef_df[coef_df["feature"] != "const"].copy()
    plot_df["label"] = plot_df["feature"].map(FEAT_LABELS)
    plot_df = plot_df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(plot_df) + 1.5)))
    colors = ["#CC4400" if c > 0 else "#1E6FBE" for c in plot_df["coef"]]
    ax.barh(plot_df["label"], plot_df["coef"], color=colors, edgecolor="white")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row["significant"]:
            ax.text(row["coef"] + (0.02 if row["coef"] > 0 else -0.02),
                    i, "*", va="center", fontsize=14,
                    ha="left" if row["coef"] > 0 else "right")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("標準化回帰係数（* p<0.05）")
    r2_loo_str = f"{r2_loo:.2f}" if not np.isnan(r2_loo) else "N/A"
    ax.set_title(f"{title}\nR²={ols.rsquared:.2f}, LOO-CV R²={r2_loo_str} (n={int(ols.nobs)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure保存: {out_path}")


def main():
    beef_df = build_dataset()
    print(f"ビーフ曲の動画数: {len(beef_df)}")
    print(beef_df[["video_id", "beef_id", "artist_name", "beef_type", "win_loss_side",
                   "pct_conflict", "view_efficiency"]].to_string())

    # --- (1) フル7特徴量モデル（指示書通り、ただし自由度不足を明記）---
    model_df_full = beef_df[FEATURES_FULL + [TARGET]].dropna()
    if len(model_df_full) >= 3:
        ols_full, coef_full, r2_loo_full = run_ols(model_df_full, FEATURES_FULL, "フルモデル(7特徴量)")
        coef_full.to_csv(OUT_DIR / "regression_results_full.csv", index=False)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_coefficients(coef_full, ols_full, r2_loo_full,
                           FIGURES_DIR / "fig_regression_full.png",
                           "対立語彙率の説明要因（フル7特徴量、n極小につき参考値）")
    else:
        print("[スキップ] フルモデル: 有効サンプル数が不足")

    # --- (2) 縮小2特徴量モデル（自由度確保、主たる解釈対象）---
    model_df_reduced = beef_df[FEATURES_REDUCED + [TARGET]].dropna()
    ols_reduced, coef_reduced, r2_loo_reduced = run_ols(
        model_df_reduced, FEATURES_REDUCED, "縮小モデル(2特徴量: ビーフタイプ・勝敗)")
    coef_reduced.to_csv(OUT_DIR / "regression_results_reduced.csv", index=False)
    plot_coefficients(coef_reduced, ols_reduced, r2_loo_reduced,
                       FIGURES_DIR / "fig_regression.png",
                       "対立語彙率の説明要因（重回帰分析、縮小モデル）")

    print("\n" + "=" * 60)
    print("結論用サマリ:")
    print(f"  フルモデル: n={len(model_df_full)}, 特徴量数={len(FEATURES_FULL)} "
          f"→ 自由度={len(model_df_full) - len(FEATURES_FULL) - 1}（過学習域、参考値に留める）")
    print(f"  縮小モデル: n={len(model_df_reduced)}, 特徴量数={len(FEATURES_REDUCED)} "
          f"→ 自由度={len(model_df_reduced) - len(FEATURES_REDUCED) - 1}")
    print("=" * 60)


if __name__ == "__main__":
    main()
