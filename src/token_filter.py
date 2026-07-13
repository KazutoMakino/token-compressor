import asyncio

import numpy as np
from agent_framework.ollama import OllamaEmbeddingClient

#######################################################################################
# main
#######################################################################################


async def main():
    # フィルタにクライアントを注入して初期化
    filter_pipeline = EmbeddingFilter(model="qwen3-embedding:0.6b")
    await filter_pipeline.initialize()

    # テスト用のコンテキスト
    raw_context = """
    お世話になっております。株式会社サンプルの田中です。
    先日はお忙しい中ご確認いただきありがとうございます。

    このたびは、第3四半期の売上分析レポートをご共有いたします。

    【分析サマリー】
    ・7月〜9月の総売上は前年同期比 +12.4% となりました。
    ・特に東日本リージョンでの新規顧客獲得数が顕著に増加しており、
    　オンライン広告施策の効果が数値として確認できます。
    ・一方、西日本リージョンでは物流コストの上昇により、粗利率が 2.1% 低下しています。

    【課題と対応方針】
    コスト増の主因は燃料費および外注委託費の高騰であるため、
    来四半期は内製化率の向上と複数ベンダーの競争見積もりを推進します。

    ご不明な点がございましたらお気軽にお申し付けください。
    引き続きよろしくお願いいたします。

    ---

    平素より大変お世話になっております。営業部の鈴木です。
    来週の定例ミーティングについて、以下の通りご案内いたします。

    【開催概要】
    ・日時：10月15日（火）14:00〜15:30
    ・場所：第2会議室（オンライン参加はZoomリンクをご参照ください）
    ・議題：Q4キャンペーン施策の最終確認、予算配分の見直し

    事前に添付の資料をご一読いただけますと幸いです。
    お手数をおかけしますが、よろしくお願いいたします。
    """

    print("\n- テスト用の生コンテキスト:")
    print("=" * 60)
    print(raw_context)
    print("=" * 60)

    print("\n- フィルタリング処理を実行します...")
    print("コンテキストからカット：")
    print("-" * 60)
    clean_text = await filter_pipeline.filter_noise_lines(raw_context, threshold=0.5)
    print("-" * 60)

    print("\n- クリーンアップされた最終コンテキスト:")
    print("=" * 60)
    print(clean_text)
    print("=" * 60)

    before_chars = len(raw_context)
    after_chars = len(clean_text)
    reduction_rate = (1 - after_chars / before_chars) * 100
    print(f"\n- 文字数: {before_chars} → {after_chars} ({reduction_rate:.1f}% 削減)")


#######################################################################################
# class
#######################################################################################


class EmbeddingFilter:
    """A high-performance noise filter that uses OllamaEmbeddingClient to remove boilerplate text."""

    def __init__(self, model: str):
        """Initialize the embedding filter with a specific model.

        Args:
            model (str): The name of the embedding model to use.
        """
        self.client = OllamaEmbeddingClient(model=model)

        # 排除したい明確な定型ノイズの基準文
        # - 過去のデータから有益なものを抽出し、ここ入れ込む（実際は別ファイルに保存して管理）
        self.noise_texts = [
            # 挨拶・書き出し
            "お世話になっております。",
            "平素より大変お世話になっております。",
            "いつもお世話になっております。",
            "はじめまして。",
            "ご連絡いただきありがとうございます。",
            # 感謝・締め
            "お忙しい中ご確認いただきありがとうございます。",
            "よろしくお願いいたします。",
            "何卒よろしくお願い申し上げます。",
            "引き続きよろしくお願いいたします。",
            "お疲れ様です。",
            "お手数をおかけしますが、よろしくお願いいたします。",
            # 定型前置き
            "以上、ご確認のほどよろしくお願いいたします。",
            "取り急ぎご連絡まで。",
            "ご不明な点がございましたらお気軽にお申し付けください。",
            "お時間をいただきありがとうございます。",
        ]
        self.noise_embeddings = []

    async def initialize(self) -> None:
        """Asynchronously initialize and cache the embeddings of reference noise texts."""
        result = await self.client.get_embeddings(self.noise_texts)
        self.noise_embeddings = [item.vector for item in result]

    def _cos_sim(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors using NumPy.

        Args:
            v1 (list[float]): The first vector.
            v2 (list[float]): The second vector.

        Returns:
            float: The cosine similarity score.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    async def filter_noise_lines(self, text: str, threshold: float = 0.5) -> str:
        """Decompose the input context into lines and remove noise based on vector similarity.

        Args:
            text (str): The input text context containing lines to be filtered.
            threshold (float, optional): The similarity threshold for noise filtering.
                Lines with a similarity score greater than or equal to this value
                will be removed. Defaults to 0.5.

        Returns:
            str: The filtered text with noise lines removed, joined by newlines.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return ""

        # 入力されたコンテキストの全行をバッチ（一括）でエンコード
        result = await self.client.get_embeddings(lines)
        line_embeddings = [item.vector for item in result]

        filtered_lines = []

        for line, line_emb in zip(lines, line_embeddings):
            # 事前にキャッシュしたノイズ群との類似度を算出
            scores = [self._cos_sim(line_emb, noise_emb) for noise_emb in self.noise_embeddings]
            max_noise_score = max(scores) if scores else 0.0

            if max_noise_score < threshold:
                filtered_lines.append(line)
            else:
                print(f"  - {line} (Score: {max_noise_score:.2f})")

        return "\n".join(filtered_lines)


#######################################################################################

if __name__ == "__main__":
    asyncio.run(main())
    exit()
