"""
build_llm_stance_sample.py (v2 追加モジュール、Task5準備)

ANTHROPIC_API_KEYがこの環境に設定されていないため、Claude APIを直接叩く代わりに
「別のClaude会話にそのまま貼り付ければ分類結果が返ってくる」形式のプロンプトファイルを
生成する。ユーザーはこのファイルの中身をコピペ、またはファイルごとアップロードするだけでよい。

各ビーフのbeefカテゴリコメントから最大250件をサンプリングし、
1つのMarkdownファイル(outputs/llm_stance_prompt.md)にまとめる。

使い方:
  python build_llm_stance_sample.py
"""

from pathlib import Path

import pandas as pd

BEEF_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = BEEF_ROOT / "data" / "processed" / "comments_clean_v2.csv"
OUT_PATH = BEEF_ROOT / "outputs" / "llm_stance_prompt.md"
SAMPLE_CSV = BEEF_ROOT / "data" / "processed" / "llm_stance_sample.csv"

N_PER_BEEF = 250
RANDOM_STATE = 42

BEEF_INFO = {
    "b1": ("Kendrick Lamar", "Drake"),
    "b2": ("Pusha T", "Drake"),
    "b4": ("Eminem", "MGK"),
    "b5": ("Lil Wayne", "Birdman"),
    "b6": ("Megan Thee Stallion", "Nicki Minaj"),
}


def main():
    df = pd.read_csv(IN_PATH, keep_default_na=False, na_values=[""])
    beef_only = df[df["track_category"] == "beef"].copy()

    samples = []
    for bid, (a, b) in BEEF_INFO.items():
        sub = beef_only[beef_only["beef_id"] == bid]
        n = min(N_PER_BEEF, len(sub))
        sample = sub.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
        sample["artist_a"] = a
        sample["artist_b"] = b
        samples.append(sample)

    all_samples = pd.concat(samples, ignore_index=True)
    all_samples["sample_id"] = range(1, len(all_samples) + 1)
    all_samples.to_csv(SAMPLE_CSV, index=False)
    print(f"サンプル抽出: {len(all_samples)}件 -> {SAMPLE_CSV}")

    # --- LLMに貼り付けるプロンプトファイルを生成 ---
    lines = []
    lines.append("# LLMスタンス分類タスク\n")
    lines.append(
        "以下はHIPHOPビーフに関するYouTubeコメントです。各コメントについて、"
        "コメントがどちらのアーティストを支持しているか（stance）を分類してください。\n"
    )
    lines.append(
        "- `a`: 1人目のアーティスト（artist_a）を支持\n"
        "- `b`: 2人目のアーティスト（artist_b）を支持\n"
        "- `neutral`: どちらでもない、中立、無関係\n"
    )
    lines.append(
        "\n**出力形式**: 他の説明文は一切含めず、`sample_id,stance` の形式のCSVのみを"
        "出力してください（ヘッダー行`sample_id,stance`を含む）。全件分を1つのコードブロックに"
        "まとめて出力してください。\n"
    )

    for bid, (a, b) in BEEF_INFO.items():
        sub = all_samples[all_samples["beef_id"] == bid]
        if len(sub) == 0:
            continue
        lines.append(f"\n## {bid}: {a} vs {b} （artist_a={a}, artist_b={b}, {len(sub)}件）\n")
        for _, row in sub.iterrows():
            text = str(row["text_clean"]).replace("\n", " ").strip()
            lines.append(f"- [{row['sample_id']}] {text}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"プロンプトファイル生成: {OUT_PATH}")
    print(f"合計サンプル数: {len(all_samples)}件（ビーフごとに最大{N_PER_BEEF}件）")
    print("\n使い方: このファイルの中身を新しいClaude会話にコピペ、またはファイルごと")
    print("アップロードしてください。返ってきたCSVを data/processed/llm_stance_result.csv")
    print("として保存すれば、merge_llm_stance_result.py（別途）で本体データと結合できます。")


if __name__ == "__main__":
    main()
