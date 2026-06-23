"""
verify_keywords.py
=================================================================
comments_topic_labeled.csvから、各キーワードの実際の使用例を確認して、
誤検出がないかを検証するスクリプト。

ユーザーが指摘した問題:
- 「style」は「ラップのスタイル/フロウ」の意味で、ファッション消費ではない
- 「aesthetic」「aura」も同様に曖昧
- 各キーワードの実際の使用例を確認して、正しいキーワードだけで再計算する

使い方:
  python verify_keywords.py
=================================================================
"""

import pandas as pd
import re

df = pd.read_csv("output_v5/data/processed/comments_topic_labeled.csv")

print("=" * 80)
print("キーワード検証：各トピックの実際のコメント例を確認")
print("=" * 80)

# v5で使用されている正規表現パターン
LIVE_RE     = re.compile(r"\b(concert|live show|live performance|tour|on stage|in concert)\b", re.IGNORECASE)
MEMORIAL_RE = re.compile(r"\b(rip|rest in peace|condolences)\b", re.IGNORECASE)
RIVALRY_RE  = re.compile(r"\b(beef|diss|versus|feud|drake|kendrick|cole|lamar)\b", re.IGNORECASE)
SCANDAL_RE  = re.compile(r"\b(epstein|diddy|scandal|expose|cancel(led)?|allegations)\b", re.IGNORECASE)

# v3で使用されていたキーワード（古い版）
# 実は検証スクリプトから見ると、v5で「style」「aesthetic」「aura」は
# キーワードとして使用されていない。つまり、ユーザーが見ているのは
# v3の結果からの言及かもしれない。

# 各トピックについて、実際に検出されたコメント数を確認
print("\n【STEP 1】各トピック別のコメント分布")
print("-" * 80)
topic_dist = df[~df['is_noise']]['topic'].value_counts()
for topic, count in topic_dist.items():
    print(f"{topic:20s}: {count:6d}件 ({count/len(df[~df['is_noise']])*100:5.2f}%)")

# 各トピックについて、サンプルコメントを表示
print("\n【STEP 2】各トピックのサンプルコメント（誤検出チェック用）")
print("=" * 80)

topics_to_check = ["⚔️ ビーフ", "🎪 ライブ", "🕯️ 追悼", "🔥 炎上"]

for topic in topics_to_check:
    sample = df[(df['topic'] == topic) & (~df['is_noise'])].head(10)
    print(f"\n【{topic}】サンプル {len(sample)}件")
    print("-" * 80)
    if len(sample) == 0:
        print("  （該当なし）")
        continue
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"{i}. [{row['artist']}] {row['text'][:100]}")
        # マッチしたキーワードを特定
        if topic == "⚔️ ビーフ" and RIVALRY_RE.search(row['text']):
            match = RIVALRY_RE.search(row['text']).group()
            print(f"   -> マッチキーワード: '{match}'")
        elif topic == "🎪 ライブ" and LIVE_RE.search(row['text']):
            match = LIVE_RE.search(row['text']).group()
            print(f"   -> マッチキーワード: '{match}'")
        elif topic == "🕯️ 追悼" and MEMORIAL_RE.search(row['text']):
            match = MEMORIAL_RE.search(row['text']).group()
            print(f"   -> マッチキーワード: '{match}'")
        elif topic == "🔥 炎上" and SCANDAL_RE.search(row['text']):
            match = SCANDAL_RE.search(row['text']).group()
            print(f"   -> マッチキーワード: '{match}'")

# 【重要】ユーザーが指摘した「style」「aesthetic」「aura」について確認
print("\n【STEP 3】ユーザー指摘のキーワードを手動で検索")
print("=" * 80)

keywords_to_check = {
    "style": "ラップのスタイル/フロウ vs ファッション",
    "aesthetic": "映像美（芸術） vs その他",
    "aura": "Dark Aura曲名 vs その他",
}

for keyword, context in keywords_to_check.items():
    pattern = re.compile(rf"\b{keyword}\b", re.IGNORECASE)
    matching = df[df['text_clean'].str.contains(keyword, case=False, na=False)]
    print(f"\n【{keyword}】{context}")
    print(f"  マッチ件数: {len(matching)}")
    if len(matching) > 0:
        print("  サンプル:")
        for i, (_, row) in enumerate(matching.head(5).iterrows(), 1):
            print(f"    {i}. [{row['artist']}] {row['text'][:80]}")

print("\n" + "=" * 80)
print("【完了】コメントを確認して、誤検出しているキーワードを特定してください。")
print("=" * 80)
