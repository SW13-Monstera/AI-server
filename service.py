# Todo : runner, service 정의
from __future__ import annotations
from schema import UserAnswer, KeywordPredictData
import bentoml
from bentoml.io import Text, JSON
from sklearn.metrics.pairwise import cosine_similarity

keyword_model = bentoml.pytorch.get("sentence-ko-roberta")


class KeywordPredictRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True
    threshold = 0.3
    word_concat_size = 3

    def __init__(self):
        self.model = bentoml.pytorch.load_model(keyword_model)

    @bentoml.Runnable.method(batchable=False)
    def is_correct_keyword(self, input_data: KeywordPredictData) -> dict:
        split_answer = input_data.user_answer.strip().split()
        tokenized_answer = []
        for k in range(len(split_answer) - self.word_concat_size + 1):
            tokenized_answer.append(' '.join(split_answer[k: k + self.word_concat_size]))
        if len(split_answer) < self.word_concat_size:
            tokenized_answer.append(' '.join(split_answer))
        tokenized_answer_embedding = self.model.encode(tokenized_answer)
        similarity_scores = cosine_similarity(input_data.embedded_keywords, tokenized_answer_embedding)
        predicts = []
        for z, idx in enumerate(similarity_scores.argmax(axis=1)):
            if self.threshold < similarity_scores[z][idx]:
                predicts.append(1)
            else:
                predicts.append(0)
        # Todo : predicts가 1인 친구 인덱스 찾아서 return 하기
        return {}


keyword_runner = bentoml.Runner(KeywordPredictRunnable, models=[keyword_model])
keyword_service = bentoml.Service(name="keyword_service", runners=[keyword_runner])


@keyword_service.api(input=JSON(pydantic_model=UserAnswer), output=JSON())
async def keyword_predict(input_data: UserAnswer) -> None:
    keywords = []  # Todo
    embedded_keywords = []  # Todo
    # 현재 메모리에 problem 정보가 있는지 확인
    # 없으면 백엔드에 요청후 키워드 임베딩해서 메모리에 저장

    return keyword_runner.is_correct_keyword.run(
        KeywordPredictData(
            problem_id=input_data.problem_id,
            user_answer=input_data.user_answer,
            keywords=keywords,
            embedded_keywords=embedded_keywords
        )
    )
