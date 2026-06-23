"""
analyze_tyler_timeline.py
=================================================================
Tyler, The Creator の時系列比較分析

目的：
  Yonkers（過激な無名期）vs 既存3曲（成熟期）
  → 商業化/成功がコメントをどう変えたか検証

手法：
  1. キーワード別ヒット件数の比較
  2. 懐古言及の検出
  3. 衝撃性・過激性言及の頻度比較
  4. サンプルコメント表示
=================================================================
"""

import pandas as pd
import re
from collections import Counter

# ============================================================
# データ読み込み
# ============================================================
print("=" * 70)
print("Tyler, The Creator 時系列比較分析")
print("=" * 70)

df_yonkers = pd.read_csv("output/data/raw/Tyler_extra_yonkers.csv")
df_mature = pd.read_csv("output/data/raw/Tyler_The_Creator.csv")

print(f"\n[データ要約]")
print(f"  Yonkers（2011年, 過激な無名期）: {len(df_yonkers)}件")
print(f"  既存3曲（2017-2019年, 成熟期）: {len(df_mature)}件")
print(f"  合計: {len(df_yonkers) + len(df_mature)}件")

# ============================================================
# キーワード定義
# ============================================================
keywords = {
    "懐古トーン": [
        r"\b(old|back|days|then)\b",
        r"(wasn't like this|used to be|miss those)",
        r"(2010|2011|2012|2013|goblin|bastard)",
    ],
    "衝撃性・過激性": [
        r"\b(shocking|crazy|wild|insane|mad)\b",
        r"\b(edgy|raw|dark|angry|aggressive)\b",
        r"\b(impact|changed|blew|shocked)\b",
    ],
    "商業化言及": [
        r"\b(commercial|mainstream|sell|out|corporate)\b",
        r"\b(grammy|award|fame|popular|success)\b",
        r"\b(change|different|evolved|matured)\b",
    ],
    "称賛（一般）": [
        r"\b(love|fire|hard|best|great|good|amazing)\b",
        r"\b(goat|legend|king)\b",
    ],
    "制作・音楽的評価": [
        r"\b(beat|production|sample|flow|lyrics|lyricism)\b",
        r"\b(creative|artistic|original)\b",
    ],
}

# ============================================================
# キーワード検索関数
# ============================================================
def search_keywords(texts, keywords_dict):
    """テキストリストからキーワード出現を数える"""
    results = {}
    for cat, patterns in keywords_dict.items():
        count = 0
        for text in texts:
            text_lower = str(text).lower()
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    count += 1
                    break  # 1つのテキストで複数パターンマッチしても1回だけカウント
        results[cat] = count
    return results

# ============================================================
# 分析実行
# ============================================================
print("\n" + "=" * 70)
print("キーワード別比較")
print("=" * 70)

yonkers_keywords = search_keywords(df_yonkers["text"].tolist(), keywords)
mature_keywords = search_keywords(df_mature["text"].tolist(), keywords)

print(f"\n{'カテゴリ':<20} {'Yonkers':<20} {'既存3曲':<20} {'差分':<10}")
print("-" * 70)

for cat in keywords.keys():
    y_count = yonkers_keywords[cat]
    m_count = mature_keywords[cat]
    y_pct = (y_count / len(df_yonkers)) * 100 if len(df_yonkers) > 0 else 0
    m_pct = (m_count / len(df_mature)) * 100 if len(df_mature) > 0 else 0
    diff = y_pct - m_pct
    
    print(f"{cat:<20} {y_count:>3}件({y_pct:>5.1f}%) {m_count:>3}件({m_pct:>5.1f}%) {diff:>+6.1f}%")

# ============================================================
# 懐古言及の詳細分析
# ============================================================
print("\n" + "=" * 70)
print("懐古言及（「あの頃のTylerが良かった」）の分析")
print("=" * 70)

nostalgia_pattern = r"(old|back|miss|remember|better|used to|back then|those days|2011|2012|2013|goblin|bastard)"

yonkers_nostalgia = df_yonkers[df_yonkers["text"].str.lower().str.contains(nostalgia_pattern, na=False)]
mature_nostalgia = df_mature[df_mature["text"].str.lower().str.contains(nostalgia_pattern, na=False)]

print(f"\nYonkers での懐古言及: {len(yonkers_nostalgia)}件 ({len(yonkers_nostalgia)/len(df_yonkers)*100:.1f}%)")
print(f"既存3曲での懐古言及: {len(mature_nostalgia)}件 ({len(mature_nostalgia)/len(df_mature)*100:.1f}%)")

print("\n[Yonkers での懐古言及サンプル（上位3件）]")
for i, (idx, row) in enumerate(yonkers_nostalgia.head(3).iterrows()):
    print(f"\n{i+1}. {row['text'][:100]}...")

print("\n[既存3曲での懐古言及サンプル（上位3件）]")
for i, (idx, row) in enumerate(mature_nostalgia.head(3).iterrows()):
    print(f"\n{i+1}. {row['text'][:100]}...")

# ============================================================
# 衝撃性への言及の詳細分析
# ============================================================
print("\n" + "=" * 70)
print("衝撃性・過激性への言及の分析")
print("=" * 70)

shock_pattern = r"(shocking|crazy|wild|insane|edgy|raw|dark|angry|aggressive|mad|impact|changed|blew)"

yonkers_shock = df_yonkers[df_yonkers["text"].str.lower().str.contains(shock_pattern, na=False)]
mature_shock = df_mature[df_mature["text"].str.lower().str.contains(shock_pattern, na=False)]

print(f"\nYonkers での衝撃性言及: {len(yonkers_shock)}件 ({len(yonkers_shock)/len(df_yonkers)*100:.1f}%)")
print(f"既存3曲での衝撃性言及: {len(mature_shock)}件 ({len(mature_shock)/len(df_mature)*100:.1f}%)")

print("\n[Yonkers での衝撃性言及サンプル]")
for i, (idx, row) in enumerate(yonkers_shock.head(3).iterrows()):
    print(f"\n{i+1}. {row['text'][:100]}...")

# ============================================================
# 考察
# ============================================================
print("\n" + "=" * 70)
print("考察：商業化によるコメント内容の変化")
print("=" * 70)

print(f"""
【発見】

1. 懐古言及の差 
   → Yonkers: {len(yonkers_nostalgia)/len(df_yonkers)*100:.1f}% vs 既存: {len(mature_nostalgia)/len(df_mature)*100:.1f}%
   
   解釈：
   - 既存3曲で懐古言及が多い場合
     → ファンは「あの頃のTylerが良かった」と現在と過去を比較している
     → 商業化・成熟によって、ファンが「失ったもの」を言及するようになった
   
   - Yonkersで懐古言及が多い場合
     → コメント欄に「あの頃に戻ってほしい」というリクエストが集中している
     → ファンは成熟期の商業的成功に、初期の「危険性」を喪失したと感じている

2. 衝撃性言及の差
   → Yonkers: {len(yonkers_shock)/len(df_yonkers)*100:.1f}% vs 既存: {len(mature_shock)/len(df_mature)*100:.1f}%
   
   解釈：
   - 差が大きい場合
     → 初期 Tyler は「衝撃」「危険」の記号だった
     → 成熟期 Tyler は「高度な芸術性」「成功」の記号に変わった
   
3. 商業化言及
   → {mature_keywords['商業化言及']} vs {yonkers_keywords['商業化言及']}
   
   解釈：
   - 既存曲で商業化言及が多い
     → ファンは明示的に「商業化」を意識している
   - 多くない
     → コメント欄では表面的な「称賛」が支配的で、
       深い批評（商業化批判）は稀である

【本質的な問い】

v3最終版レポートで見えた「出来事駆動型」コメント（訃報・ビーフ・炎上）
に対し、今回の時系列比較は、より根本的な問い：

「商業的成功がアーティストのイメージそのものを再構成し、
 ファンの言及内容を『危険性への言及』から『成功の確認』へと
 シフトさせるのか？」

を検証している。
""")

# ============================================================
# サンプルコメント出力
# ============================================================
print("\n" + "=" * 70)
print("無作為サンプルコメント（内容確認用）")
print("=" * 70)

print("\n[Yonkers ランダムサンプル 5件]")
for i, (idx, row) in enumerate(df_yonkers.sample(5, random_state=42).iterrows()):
    likes = row.get("likes", 0)
    print(f"\n{i+1}. (❤️ {likes}件) {row['text'][:120]}")

print("\n[既存3曲 ランダムサンプル 5件]")
for i, (idx, row) in enumerate(df_mature.sample(5, random_state=42).iterrows()):
    likes = row.get("likes", 0)
    print(f"\n{i+1}. (❤️ {likes}件) {row['text'][:120]}")

print("\n" + "=" * 70)
print("分析完了")
print("=" * 70)
