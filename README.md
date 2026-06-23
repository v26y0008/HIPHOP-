# データ取得手順（急ぎ用）

## 0. 事前準備
```
pip install -r requirements.txt
```

YouTube Data API v3 のAPIキーを取得:
https://console.cloud.google.com/ → プロジェクト作成 → 「APIとサービス」→
「YouTube Data API v3」を有効化 → 「認証情報」→ APIキーを作成

APIキーを環境変数に設定:
```
export YOUTUBE_API_KEY="ここにAPIキー"
```
(Windowsの場合: `set YOUTUBE_API_KEY=ここにAPIキー`)

## 1. 動画IDを探す（各アーティストごとに実行）
```
python find_video_ids.py "Nas" "official video"
python find_video_ids.py "Joey Bada$$" "official video"
python find_video_ids.py "GZA" "official video"
python find_video_ids.py "The Roots" "official video"
python find_video_ids.py "Future" "official video"
python find_video_ids.py "Migos" "official video"
python find_video_ids.py "Gucci Mane" "official video"
python find_video_ids.py "21 Savage" "official video"
python find_video_ids.py "Tyler The Creator" "official video"
python find_video_ids.py "JPEGMAFIA" "official video"
python find_video_ids.py "Earl Sweatshirt" "official video"
python find_video_ids.py "Death Grips" "official video"
```

表示された一覧から、**チャンネル名が本人の公式チャンネルになっているもの**を
3本ずつ選ぶ（ライブ・コンサート・インタビューは除外）。

⚠ 旧データで "Nas" が Nelly / Lil Nas X の動画と誤って紐づいた事故があったため、
   チャンネル名を必ず目で確認すること。

## 2. fetch_youtube_comments.py を編集
`VIDEO_LIST` の中の `"REPLACE_ME_1"` 等を、1で見つけた実際の video_id に書き換える。

## 3. 実行
```
python fetch_youtube_comments.py
```

→ `all_comments_new.csv` が生成されます。これを次の分析ステップ（前処理・クラスタリング）に渡してください。

## 出力ファイルの列
| 列名 | 内容 |
|---|---|
| artist | アーティスト名 |
| genre | ジャンル（Boom bap / Lyric, Trap, Alternative） |
| video_id | YouTube動画ID |
| video_title | 動画タイトル |
| channel_title | 投稿チャンネル名（検証用） |
| text | コメント本文 |
| likes | いいね数 |
