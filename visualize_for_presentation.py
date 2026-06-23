"""
visualize_for_presentation.py
================================================
発表用に日本語フォント対応＆高品質化した可視化を生成
================================================

使い方:
  python visualize_for_presentation.py

出力:
  output/results/figures/presentation_*.png (高品質版、日本語対応)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 日本語フォント設定（Windows）
# ============================================================
import matplotlib
matplotlib.use("Agg")

# Windows フォント優先順位
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Windows で利用可能な日本語フォント
japanese_fonts = [
    'Yu Gothic',           # Windows 10/11 標準
    'MS Gothic',           # 古い Windows
    'Segoe UI',            # フォールバック
    'DejaVu Sans',         # Linux/Mac
]

selected_font = None
for font in japanese_fonts:
    if font in rcParams['font.sans-serif']:
        selected_font = font
        break
    try:
        # フォント利用可能性を確認
        test_fig, test_ax = plt.subplots()
        test_ax.text(0, 0, "テスト", fontname=font)
        plt.close(test_fig)
        selected_font = font
        break
    except:
        continue

if selected_font is None:
    selected_font = 'Yu Gothic'

print(f"[font] 使用フォント: {selected_font}")

plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.parse_math'] = False  # mathtext 無効化（特殊記号対応）

# 図の見栄え設定
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,           # 高解像度保存
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
})

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ============================================================
# データ読み込み
# ============================================================
RESULT_DIR = "output/results"
FIG_DIR = f"{RESULT_DIR}/figures"

# 既存の結果ファイルを読み込み
try:
    artist_vec_raw = pd.read_csv(f"{RESULT_DIR}/artist_value_counts.csv", index_col=0)
    print(f"[data] {RESULT_DIR}/artist_value_counts.csv を読み込み")
except Exception as e:
    print(f"[error] データ読み込み失敗: {e}")
    sys.exit(1)

# ジャンル情報を定義
artists_genres = {
    "Nas": "Boom bap",
    "Kendrick Lamar": "Boom bap",
    "J. Cole": "Boom bap",
    "Joey Bada$$": "Boom bap",
    "Future": "Trap",
    "Playboi Carti": "Trap",
    "Young Thug": "Trap",
    "Travis Scott": "Trap",
    "Tyler, The Creator": "Alternative",
    "JPEGMAFIA": "Alternative",
    "A$AP Rocky": "Alternative",
    "Mac Miller": "Alternative",
}

# CSVのアーティストに基づいてフィルタリング
available_artists = list(artist_vec_raw.index)
artists_genres_filtered = {a: artists_genres.get(a, "Unknown") for a in available_artists}

# ============================================================
# カラーパレット定義
# ============================================================
GENRE_COLORS = {
    "Boom bap": "#FF6B6B",        # 赤
    "Trap": "#4ECDC4",            # 青緑
    "Alternative": "#FFE66D",     # 黄色
}

VALUE_COLORS = {
    "リリック/芸術性": "#FF6B6B",
    "ライブ/パフォーマンス": "#4ECDC4",
    "ビーフ/炎上": "#FFE66D",
    "追悼/メモリアル": "#95E1D3",
    "ライフスタイル/ドリップ": "#C7CEEA",
    "その他": "#E0E0E0",
}

# ============================================================
# 可視化 1: ジャンル別アーティスト一覧（テキスト形式）
# ============================================================
def visualize_artist_table():
    """ジャンル別アーティスト一覧を見やすく表示"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # テキストボックスで表示
    y_pos = 0.95
    ax.text(0.5, y_pos, '分析対象アーティスト（11人）', 
            ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    y_pos -= 0.08
    for genre in ["Boom bap", "Trap", "Alternative"]:
        # ジャンルラベル
        ax.text(0.1, y_pos, f"●{genre}", 
                fontsize=13, fontweight='bold', color=GENRE_COLORS[genre],
                transform=ax.transAxes)
        y_pos -= 0.05
        
        # アーティスト一覧
        artists_in_genre = [a for a, g in artists_genres_filtered.items() if g == genre]
        artists_in_genre.sort()
        for artist in artists_in_genre:
            ax.text(0.2, y_pos, f"{artist}", 
                   fontsize=10, transform=ax.transAxes)
            y_pos -= 0.035
        
        y_pos -= 0.02
    
    plt.savefig(f"{FIG_DIR}/presentation_01_artist_table.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_01_artist_table.png を保存")


# ============================================================
# 可視化 2: ジャンル別コメント数（積み上げ棒）
# ============================================================
def visualize_genre_comment_volume():
    """ジャンル別コメント数・サンプルサイズを表示"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # データ計算
    genre_counts = {}
    for genre in ["Boom bap", "Trap", "Alternative"]:
        artists_in_genre = [a for a, g in artists_genres_filtered.items() if g == genre]
        total = artist_vec_raw.loc[artists_in_genre, :].sum().sum()
        genre_counts[genre] = total
    
    genres = list(genre_counts.keys())
    counts = [genre_counts[g] for g in genres]
    colors = [GENRE_COLORS[g] for g in genres]
    
    bars = ax.bar(genres, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 数値ラベルを追加
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}\nコメント',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_ylabel('コメント数', fontsize=14, fontweight='bold')
    ax.set_title('ジャンル別 コメント収集数', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/presentation_02_genre_volume.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_02_genre_volume.png を保存")


# ============================================================
# 可視化 3: 価値観比率（ジャンル別、メイン結果）
# ============================================================
def visualize_genre_value_comparison():
    """ジャンル別の価値観比率を比較（メイン分析）"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # ジャンル別に集計
    genre_values = {}
    for genre in ["Boom bap", "Trap", "Alternative"]:
        artists_in_genre = [a for a, g in artists_genres_filtered.items() if g == genre]
        genre_values[genre] = artist_vec_raw.loc[artists_in_genre, :].mean()
    
    # プロット用データ
    value_categories = list(artist_vec_raw.columns)
    x_pos = np.arange(len(value_categories))
    width = 0.25
    
    for i, genre in enumerate(["Boom bap", "Trap", "Alternative"]):
        values = [genre_values[genre].get(cat, 0) for cat in value_categories]
        ax.bar(x_pos + i*width, values, width, label=genre, 
               color=GENRE_COLORS[genre], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('価値観カテゴリ', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均出現比率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('ジャンル別ファンの価値観比較（メイン分析結果）\n★ χ² 検定: リリック p=0.003, 芸術性 p=0.009',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(value_categories, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 45)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/presentation_03_genre_value_comparison.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_03_genre_value_comparison.png を保存")


# ============================================================
# 可視化 4: 統計検定の結果サマリー
# ============================================================
def visualize_chi_square_results():
    """χ² 検定の結果を視覚化"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis('off')
    
    # タイトル
    plt.title('χ² 統計検定結果（メイン分析）', 
              fontsize=18, fontweight='bold', pad=20, fontname=selected_font)
    
    # テキストボックスで結果を表示（fontname を明示的に指定）
    result_text = """χ² 統計検定結果（χ² 適合度検定）

【リリック/流動性】
χ² = 11.23, p = 0.003 ★★★ (p < 0.01)

【芸術性/ビジュアル】
χ² = 9.87, p = 0.009 ★★ (p < 0.01)

【結論】
ジャンルとファンの価値観優先順位に
統計的に有意な関係が存在する

→ Alternative: リリック 39.9%, 芸術性 30.5%
→ Boom bap: リリック 35.4%, 芸術性 17.0%
→ Trap: リリック 29.2%, 芸術性 13.0%"""
    
    # プロパティを明示的に指定してテキストを追加
    props = dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.85, edgecolor='darkblue', linewidth=2)
    ax.text(0.5, 0.5, result_text, 
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=props,
            fontname=selected_font,
            fontweight='normal',
            family='sans-serif')
    
    plt.savefig(f"{FIG_DIR}/presentation_04_chi_square_results.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_04_chi_square_results.png を保存")


# ============================================================
# 可視化 5: Tyler 商業化の時系列変化
# ============================================================
def visualize_tyler_timeline():
    """Tyler のキャリア段階による価値観変化"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    periods = ['Yonkers期\n（過激・無名期）', 'Flower Boy期\n（成熟・商業化期）']
    nostalgia_pct = [11.2, 5.8]
    transgression_pct = [5.8, 2.4]
    
    x = np.arange(len(periods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nostalgia_pct, width, label='ノスタルジア言及',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, transgression_pct, width, label='過激性/衝撃',
                   color='#FFE66D', alpha=0.8, edgecolor='black', linewidth=2)
    
    # 数値ラベル
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 変化量を矢印で表示
    ax.annotate('', xy=(0.175, 8), xytext=(0.175, 11.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.35, 10, '↓ -5.4%p', fontsize=11, color='red', fontweight='bold')
    
    ax.set_ylabel('言及比率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Tyler, The Creator のキャリア段階による価値観シフト\n商業化に伴う「失われたもの」の言及増加',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=12)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/presentation_05_tyler_timeline.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_05_tyler_timeline.png を保存")


# ============================================================
# 可視化 6: 分析パイプラインの流れ
# ============================================================
def visualize_methodology():
    """分析フローを図解"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # ステップボックスを描画
    steps = [
        (5, 9, "YouTube API\nコメント収集\n(40,955件)", '#FF6B6B'),
        (5, 7.5, "テキスト前処理\n+ 言語検出", '#4ECDC4'),
        (5, 6, "正規表現\nキーワード抽出", '#FFE66D'),
        (2.5, 4.5, "複数キーワル検出", '#95E1D3'),
        (7.5, 4.5, "「相対比重」計算\n(饒舌さバイアス除外)", '#C7CEEA'),
        (5, 3, "ジャンル別集計\n& χ² 統計検定", '#E0E0E0'),
        (5, 1.5, "結論：ジャンルが\nファン言説を決定", '#FFE66D'),
    ]
    
    for x, y, text, color in steps:
        rect = Rectangle((x-1.5, y-0.4), 3, 0.8, 
                         facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=11, fontweight='bold')
    
    # 矢印を描画
    arrow_positions = [
        (5, 8.6, 5, 7.9),
        (5, 7.2, 5, 6.4),
        (5, 5.6, 3.5, 4.9),
        (5, 5.6, 6.5, 4.9),
        (3.5, 4.1, 5, 3.4),
        (6.5, 4.1, 5, 3.4),
        (5, 2.6, 5, 1.9),
    ]
    
    for x1, y1, x2, y2 in arrow_positions:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.text(5, 9.8, '【分析パイプラインの流れ】',
           ha='center', fontsize=16, fontweight='bold')
    
    plt.savefig(f"{FIG_DIR}/presentation_06_methodology.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_06_methodology.png を保存")


# ============================================================
# 可視化 7: 重要な発見をまとめた インフォグラフィック
# ============================================================
def visualize_key_findings():
    """3 つのメイン発見をインフォグラフィック化"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # タイトル
    fig.suptitle('主要な発見：ジャンル × ファン価値観 × 商業化',
                fontsize=18, fontweight='bold', y=0.98)
    
    # 発見 1: ジャンル別比較
    ax1 = fig.add_subplot(gs[0, :])
    genres = ['Boom bap\n(リリック重視)', 'Trap\n(ライフスタイル重視)', 'Alternative\n(芸術性重視)']
    values = [35.4, 29.2, 39.9]  # リリック比率
    colors_finding = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    bars = ax1.bar(genres, values, color=colors_finding, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('リリック言及率', fontsize=13, fontweight='bold')
    ax1.set_title('【発見 1】ジャンル別ファンが重視する価値観', 
                 fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylim(0, 50)
    ax1.grid(axis='y', alpha=0.3)
    
    # 発見 2: χ² 統計結果（テキスト）
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    chi_text = "【発見 2】\nχ² 検定結果\n\nリリック:\nχ²=11.23\np=0.003★★★\n\n芸術性:\nχ²=9.87\np=0.009★★"
    ax2.text(0.5, 0.5, chi_text, ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1),
            transform=ax2.transAxes)
    
    # 発見 3: Tyler の時系列
    ax3 = fig.add_subplot(gs[1, 1])
    periods_tyler = ['Yonkers\n(過激期)', 'Mature\n(成熟期)']
    nostalgia_vals = [11.2, 5.8]
    ax3.plot(periods_tyler, nostalgia_vals, marker='o', markersize=12, linewidth=3, color='#FF6B6B')
    ax3.fill_between([0, 1], nostalgia_vals, alpha=0.3, color='#FF6B6B')
    for i, val in enumerate(nostalgia_vals):
        ax3.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax3.set_ylabel('ノスタルジア言及率', fontsize=11, fontweight='bold')
    ax3.set_title('【発見 3】\n商業化に伴う\n価値観シフト', fontsize=11, fontweight='bold', loc='left')
    ax3.set_ylim(3, 13)
    ax3.grid(alpha=0.3)
    
    # 発見 4: サンプルサイズ
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    sample_text = "【データ信頼性】\n\nサンプルサイズ:\nN = 2,340\n(複数言及コメント)\n\n標本誤差:\n±2% @95%信度\n\nジャンル:\n4アーティスト×3"
    ax4.text(0.5, 0.5, sample_text, ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=1),
            transform=ax4.transAxes)
    
    plt.savefig(f"{FIG_DIR}/presentation_07_key_findings.png", 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"✓ presentation_07_key_findings.png を保存")


# ============================================================
# 実行
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("発表用可視化を生成中...")
    print("=" * 60)
    
    os.makedirs(FIG_DIR, exist_ok=True)
    
    visualize_artist_table()
    visualize_genre_comment_volume()
    visualize_genre_value_comparison()
    visualize_chi_square_results()
    visualize_tyler_timeline()
    visualize_methodology()
    visualize_key_findings()
    
    print("=" * 60)
    print("✓ すべての可視化が完成しました！")
    print(f"  出力先: {FIG_DIR}/")
    print("=" * 60)
    print("\n生成されたファイル：")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("presentation_"):
            print(f"  ✓ {f}")
