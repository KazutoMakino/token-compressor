"""src/aiai/infra/llm_judge.py"""

import re

from agent_framework import Message
from agent_framework.ollama import OllamaChatClient

# 評価用基準プロンプト
JUDGE_PROMPT_TEMPLATE = """
あなたは厳格なAI品質評価官です。
元のテキスト（Context）から抽出された情報（Clean Context）を元に、作成された回答（Response）の品質を以下の基準で評価してください。

【評価基準】
5点: 重要な情報がすべて含まれており、ハルシネーション（嘘）がなく、簡潔で完璧な回答。
4点: ほぼ正確だが、表現を少し改善できる、または不要な情報がわずかに残っている。
3点: 最低限の要満は満たしているが、重要な情報が一部抜けているか、少し不自然。
2点: 情報の欠落が多く、回答として不十分。
1点: 全く的外れな回答、または致命的な嘘（ハルシネーション）が含まれる。

【入力データ】
[Original Context]:
{raw_context}

[Clean Context]:
{clean_context}

[Response]:
{response_text}

【出力フォーマット】
思考プロセスを簡単に述べた後、必ず最後に「Score: [点数]」の形式で出力してください。例: Score: 4
"""

#######################################################################################
# class
#######################################################################################


class LLMAsAJudge:
    """A class to evaluate the quality of AI-generated text using an LLM."""

    def __init__(self, host: str, model: str):
        """Initialize the judge with Ollama client settings.

        Args:
            host (str): The host address for the Ollama service.
            model (str): The name of the LLM model to use.
        """
        self.client = OllamaChatClient(host=host, model=model)

    async def evaluate(self, raw_context: str, clean_context: str, response_text: str) -> float:
        """Evaluate the quality of a generated text based on provided context and return a score from 1.0 to 5.0.

        Args:
            raw_context (str): The original source context.
            clean_context (str): The cleaned/extracted information from the context.
            response_text (str): The AI-generated response to be evaluated.

        Returns:
            float: A score between 1.0 and 5.0 representing the quality of the response.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            raw_context=raw_context, clean_context=clean_context, response_text=response_text
        )

        messages = [Message(role="user", contents=prompt)]

        try:
            response = await self.client.get_response(messages=messages)
            judge_text = response.text
            print(f"- 評価：\n{judge_text}\n")

            # 末尾などから "Score: X" を抽出
            match = re.search(r"Score:\s*([1-5])", judge_text)
            if match:
                return float(match.group(1))

            return 1.0  # パース失敗時は最低点

        except Exception as e:
            print(f"Judge evaluation failed: {e}")
            return 1.0
