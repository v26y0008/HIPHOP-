"""
timeseries_analysis.py (v2 追加モジュール)

v1設計のPRE/PEAK/POSTは同一動画内の時期区分だったが、v2設計はcatalog/beef/postが
それぞれ別動画のため、直接の時系列DiDは行えない。代わりに以下2つを実施する:

  1. 記述統計としての時系列可視化: track_category別に、動画公開からの経過日数
     (days_since_release)でビン分けした対立語彙率の推移をビーフごとにプロット。
     「対立語彙への言及は動画公開直後に集中するのか、後々まで続くのか」を見る。
  2. catalog(疑似PRE) vs beef(疑似PEAK)の対立語彙率について、5ビーフを対とみなした
     Wilcoxon符号順位検定。個別ビーフごとのカイ二乗検定(conflict_vocab_analysis.py)
     とは異なり、5ビーフを横断して「catalogからbeefへの上昇」が一貫した方向を
     持つかを1つの検定に集約する（n=5のため検定力は低いことを明記）。

使い方:
  python timeseries_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams["font.family"] = "Meiryo"

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_with_win_loss_v2.csv"
OUT_DIR = BEEF_ROOT / "data" / "processed"
FIGURES_DIR = BEEF_ROOT / "outputs" / "figures"

BEEF_LABELS = {
    "b1": "B1: Kendrick vs Drake", "b2": "B2: Pusha T vs Drake",
    "b4": "B4: Eminem vs MGK", "b5": "B5: Wayne vs Birdman", "b6": "B6: Megan vs Nicki",
}
CAT_COLORS = {"catalog": "#999999", "beef": "#0072B2", "post": "#E69F00"}
BIN_DAYS = 7  # 週単位でビン分け
MAX_DAYS = 84  # 12週間まで


def main():
    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    df = df[df["days_since_release"] >= 0].copy()

    beef_ids = [b for b in ["b1", "b2", "b4", "b5", "b6"] if b in df["beef_id"].unique()]

    # === 1) 記述統計: 経過日数ビン別 対立語彙率の推移 ===
    fig, axes = plt.subplots(1, len(beef_ids), figsize=(3.8 * len(beef_ids), 4.2), sharey=True)
    for ax, bid in zip(axes, beef_ids):
        sub = df[df["beef_id"] == bid].copy()
        sub = sub[sub["days_since_release"] <= MAX_DAYS]
        sub["day_bin"] = (sub["days_since_release"] // BIN_DAYS) * BIN_DAYS
        for cat, color in CAT_COLORS.items():
            cat_sub = sub[sub["track_category"] == cat]
            if len(cat_sub) == 0:
                continue
            binned = cat_sub.groupby("day_bin")["has_conflict"].mean() * 100
            if len(binned) == 0:
                continue
            ax.plot(binned.index, binned.values, marker="o", markersize=3,
                    color=color, label=cat, linewidth=1.5)
        ax.set_title(BEEF_LABELS.get(bid, bid), fontsize=9)
        ax.set_xlabel("公開からの経過日数")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("対立語彙率 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=10)
    fig.suptitle(f"対立語彙率の時間推移（週次ビン、公開後{MAX_DAYS}日まで）", y=1.15, fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_timeseries_conflict_rate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure保存: {out_path}")

    # === 2) catalog(疑似PRE) vs beef(疑似PEAK) の対応ありWilcoxon検定 ===
    pre_rates, peak_rates, rows = [], [], []
    for bid in beef_ids:
        sub = df[df["beef_id"] == bid]
        pre = sub[sub["track_category"] == "catalog"]["has_conflict"].mean()
        peak = sub[sub["track_category"] == "beef"]["has_conflict"].mean()
        pre_rates.append(pre)
        peak_rates.append(peak)
        rows.append({"beef_id": bid, "catalog_rate": pre * 100, "beef_rate": peak * 100,
                      "diff_pp": (peak - pre) * 100})
        print(f"{bid}: catalog={pre*100:.1f}%  beef={peak*100:.1f}%  diff={100*(peak-pre):+.1f}pp")

    result_df = pd.DataFrame(rows)
    print("\n" + result_df.to_string(index=False))

    stat, p = stats.wilcoxon(pre_rates, peak_rates)
    print(f"\nWilcoxon符号順位検定 (catalog vs beef, n={len(beef_ids)}ビーフ対応あり): "
          f"stat={stat:.3f}, p={p:.4f}")
    print(f"5ビーフ中{sum(1 for d in result_df['diff_pp'] if d > 0)}件がcatalog→beefで上昇方向")
    print("（n=5と極小標本のため検定力は低い。個別ビーフのカイ二乗検定"
          "（conflict_vocab_analysis.py）が主たる検定、本検定はビーフ横断の補完的位置づけ）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_DIR / "timeseries_wilcoxon_result.csv", index=False)

    summary = pd.DataFrame([{
        "n_beefs": len(beef_ids), "wilcoxon_stat": stat, "p_value": p,
        "n_positive_direction": sum(1 for d in result_df["diff_pp"] if d > 0),
    }])
    summary.to_csv(OUT_DIR / "timeseries_wilcoxon_summary.csv", index=False)
    print(f"\n保存: {OUT_DIR}")


if __name__ == "__main__":
    main()
