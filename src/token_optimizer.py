"""src/token_optimizer.py"""

import asyncio

import optuna
from agent_framework import Message
from agent_framework.ollama import OllamaChatClient

from llm_judge import LLMAsAJudge
from token_filter import EmbeddingFilter

#######################################################################################
# main
#######################################################################################


def main():
    raw_context = """
    鈴木: えーっと、じゃあ時間になりましたので、今週の設備監視システムのRAG構築プロジェクトの定例、始めまーす。
    佐藤: はーい、お願いしまーす。
    鈴木: ええと、まずですね、先週問題になってた、あのー、本番インフラのセキュリティ審査の件なんですけど、あれ、法務とか情シスへの説明ロジック、固まりましたかね？高橋さん。
    高橋: あ、はい。それなんですけど、えっと、結論から言うと、データの保管先、つまり静的なドキュメントですね、PDFとか。それは「Azure Blob Storage」に置いて、会話の履歴とか、あとアカウントのメタデータみたいな動的なデータは「Azure Cosmos DB」に格納するっていう、用途別で完全に分ける設計にしました。
    鈴木: なるほど、はいはい。それって効率とかセキュリティの面では突っ込まれそうですか？
    高橋: あ、そこは大丈夫です。説明のストーリーとしては、まずBlob Storageの方はコンテナレベルで企業ごとに物理隔離します。で、Cosmos DBの方は、すべてのデータに tenant_id を持たせて、クエリのレベルで論理隔離を徹底するっていう風に、ホワイトペーパーに明記します。
    佐藤: あ、なんか、インターネットから直接アクセスできちゃうと、ハッキングとか怖いって言われませんかね？
    高橋: ああ、そうそう、それね。そこが一番のポイントで、BlobもCosmosも「プライベートエンドポイント」を使って、インターネットからのアクセスを完全に遮断します。弊社が運用するContainer Appsの仮想ネットワーク内からしかアクセスできないように、身内だけの秘密の回線で直結するので、外部漏洩リスクはゼロになります、っていうロジックで通します。
    鈴木: おお、いいですね。完璧じゃないですか。あ、ちなみに今日のランチって、みんな何食べます？私、下のセブンで弁当買おうかなって。
    佐藤: 私はちょっと、あとで駅前のラーメン屋にマシマシで食べに行こうと思ってます。
    鈴木: いいなー。あ、ごめんなさい、雑談挟んじゃった。ええと、高橋さん、今の説明を情シス向けの提案書として、エグゼクティブに見せられるような綺麗なスライド構成の箇条書きに、LLMを使って要約しておいてもらえますか？
    高橋: 了解です。じゃあ、無駄な雑談のログとか、言い淀みの部分とかは、フィルターの閾値を調整して綺麗に削ぎ落としてから、LLMに要約させますね。
    鈴木: はーい、じゃあ次のアジェンダなんですけど、えっと、なんだっけな……
    """

    optimizer = PromptOptimizer(ollama_host="http://localhost:11434", model_name="gemma4:12b-it-qat")
    optimizer.run_optimization(raw_context=raw_context, n_trials=5)


#######################################################################################
# class
#######################################################################################


class PromptOptimizer:
    """A class to optimize prompt parameters using Optuna by balancing cost and quality."""

    def __init__(self, ollama_host: str, model_name: str):
        """Initialize the optimizer with Ollama client settings.

        Args:
            ollama_host (str): The host address for the Ollama service.
            model_name (str): The name of the LLM model to use.
        """
        self.ollama_host = ollama_host
        self.model_name = model_name

    def create_objective(self, raw_context: str):
        """Create an Optuna objective function for optimization.

        Args:
            raw_context (str): The source context to be used during trials.

        Returns:
            callable: An Optuna objective function that returns a tuple of (cost, score).
        """

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            threshold = trial.suggest_float(name="filter_threshold", low=0.3, high=0.7)

            async def pipeline_run():
                # ループの競合を防ぐため、トライアル（スレッド）毎に独立してインスタンスを生成する
                noise_filter = EmbeddingFilter(model="qwen3-embedding:0.6b")
                client = OllamaChatClient(host=self.ollama_host, model=self.model_name)
                judge = LLMAsAJudge(host=self.ollama_host, model=self.model_name)

                await noise_filter.initialize()

                # 1. フィルター処理
                clean_context = await noise_filter.filter_noise_lines(raw_context, threshold=threshold)

                # 2. メインLLMでの回答生成
                messages = [
                    Message(
                        role="user",
                        contents=f"以下のコンテキストを元に、適切な回答を作成してください。\n\n{clean_context}",
                    )
                ]
                response = await client.get_response(messages=messages)
                response_text = response.text

                # 3. Judgeでの評価
                score = await judge.evaluate(
                    raw_context=raw_context,
                    clean_context=clean_context,
                    response_text=response_text,
                )

                cost = float(len(clean_context))
                return cost, score

            # 💡 各トライアル完全に独立したイベントループを立ち上げる
            new_loop = asyncio.new_event_loop()
            try:
                cost, score = new_loop.run_until_complete(pipeline_run())
                return cost, score
            finally:
                new_loop.close()

        return objective

    def run_optimization(self, raw_context: str, n_trials: int = 10):
        """Execute the optimization process and print the best results.

        Args:
            raw_context (str): The source context to be used during trials.
            n_trials (int): The number of optimization trials to run. Defaults to 10.

        Returns:
            optuna.study.Study: The completed Optuna study object.
        """
        study = optuna.create_study(directions=["minimize", "maximize"])
        objective_func = self.create_objective(raw_context)

        # 外側は純粋な同期処理として実行し、中で毎回完全に独立したイベントループを立ち上げる
        study.optimize(objective_func, n_trials=n_trials)

        print("\n🏆 【最適化完了】 パレート最適解:")
        for i, trial in enumerate(study.best_trials):
            print(f"  解 {i + 1}:")
            print(f"    パラメータ: {trial.params}")
            print(f"    評価値 [コスト(文字数), スコア]: {trial.values}")

        return study.best_trials


#######################################################################################

if __name__ == "__main__":
    main()
    exit()
