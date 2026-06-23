"""
collect_extra_tyler.py
=================================================================
Tyler, The Creator の「Yonkers」コメント追加収集

目的：
  既存データ（See You Again, EARFQUAKE, Who Dat Boy）は全て成熟期・Grammy後
  ↓
  Yonkers（Goblin期・過激な無名期）を追加して時系列比較

実行：
  python collect_extra_tyler.py

出力：
  output/data/raw/Tyler_extra_yonkers.csv（1000～1200件）
=================================================================
"""

import os
import sys
import time
import pandas as pd
from googleapiclient.discovery import build

# ============================================================
# 設定
# ============================================================
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyASkpFtAv7aUOSSjoEJ-2iJfDSvVQKeG5Q")
OUT_DIR = "output/data/raw"

def collect_tyler_yonkers():
    """Tyler, The Creator の「Yonkers」コメント収集"""
    
    print("=" * 60)
    print("Tyler, The Creator「Yonkers」コメント追加収集")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    # ============================================================
    # Step 1: Tyler公式チャンネルから「Yonkers」を検索
    # ============================================================
    print("\n[STEP 1] Tyler, The Creator チャンネル確認...")
    print("(グローバル検索を使用するためチャンネル確認は不要)\n")
    
    # ============================================================
    # Step 2: Yonkers を全体で検索（チャンネル限定ではなく）
    # ============================================================
    print("\n[STEP 2] グローバル検索で「Yonkers」動画を検索...")
    
    try:
        res_search = youtube.search().list(
            q="Tyler The Creator Yonkers Official",
            part="snippet",
            type="video",
            maxResults=5
        ).execute()
    except Exception as e:
        print(f"❌ 検索エラー: {e}")
        return None
    
    # 最初の結果を採用（通常 Yonkers 公式動画）
    if not res_search.get("items"):
        print("❌ Yonkers が見つかりません")
        return None
    
    video_item = res_search["items"][0]
    video_id = video_item["id"]["videoId"]
    video_title = video_item["snippet"]["title"]
    print(f"✅ 採用動画: 「{video_title}」")
    print(f"   Video ID: {video_id}")
    
    # 動画の詳細情報を取得
    channel_title = "Unknown"
    try:
        res_video = youtube.videos().list(
            id=video_id, part="snippet,statistics"
        ).execute()
        if res_video.get("items"):
            channel_title = res_video["items"][0]["snippet"]["channelTitle"]
            views = int(res_video["items"][0]["statistics"].get("viewCount", 0))
            print(f"   チャンネル: {channel_title}")
            print(f"   再生数: {views:,}回")
    except Exception as e:
        print(f"   ⚠️  詳細情報取得エラー: {e}")
    
    # ============================================================
    # Step 3: コメント収集
    # ============================================================
    print(f"\n[STEP 3] コメント収集中... (最大1200件)")
    
    comments = []
    token = None
    batch_count = 0
    
    while len(comments) < 1200:
        try:
            kwargs = dict(
                videoId=video_id,
                part="snippet",
                maxResults=100,
                textFormat="plainText"
            )
            if token:
                kwargs["pageToken"] = token
            
            res = youtube.commentThreads().list(**kwargs).execute()
            
            for item in res.get("items", []):
                snip = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "artist": "Tyler, The Creator",
                    "genre": "Alternative",
                    "video_id": video_id,
                    "video_title": video_title,
                    "channel_title": channel_title,
                    "period": "Goblin期（過激な無名期）",
                    "text": snip["textDisplay"],
                    "likes": snip["likeCount"]
                })
                batch_count += 1
            
            token = res.get("nextPageToken")
            if not token:
                print(f"   → {len(comments)}件取得完了（パジネーション終了）")
                break
            
            print(f"   → {len(comments)}件取得中...", end="\r")
            time.sleep(0.5)
        
        except Exception as e:
            print(f"\n⚠️  コメント取得エラー: {e}")
            break
    
    if not comments:
        print("❌ コメントが取得できませんでした")
        return None
    
    # ============================================================
    # Step 4: CSV保存
    # ============================================================
    print(f"\n[STEP 4] データ保存中...")
    
    df = pd.DataFrame(comments)
    output_path = f"{OUT_DIR}/Tyler_extra_yonkers.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ 保存完了: {output_path}")
    print(f"   → {len(df)} コメント")
    
    # ============================================================
    # Step 5: 既存データとの比較を意識
    # ============================================================
    print("\n" + "=" * 60)
    print("次のステップ（データ読み込み時）の参考情報")
    print("=" * 60)
    print(f"""
既存Tyler動画（成熟期・Grammy後）:
  - See You Again（2017年）
  - EARFQUAKE（2019年）
  - Who Dat Boy（2017年）

今回追加（過激な無名期）:
  - Yonkers（2011年, Goblin期）

比較ポイント:
  ✓ コメントの内容に「懐古トーン」はあるか
  ✓ 「あの頃のTylerが良かった」という言及の有無
  ✓ 衝撃性・過激性への言及が多いか
  ✓ 商業化について言及がないか
""")
    
    return output_path

if __name__ == "__main__":
    result = collect_tyler_yonkers()
    if result:
        print(f"\n✅ 成功: {result}")
    else:
        print("\n❌ 失敗")
        sys.exit(1)
