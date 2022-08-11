from typing import Optional

import bentoml
from bentoml.io import JSON
from sklearn.metrics.pairwise import cosine_similarity

from schemas import Keyword, KeywordInferenceData, KeywordInferenceResponse, KeywordResponse, Problem

keyword_model = bentoml.pytorch.get("sentence-ko-roberta")


class KeywordPredictRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True
    threshold = 0.3
    word_concat_size = 3

    def __init__(self, problem_dict: Optional[dict] = None):
        self.model = bentoml.pytorch.load_model(keyword_model)
        self.problem_dict = problem_dict if problem_dict else {}

    def synchronize_keywords(self, input_data: KeywordInferenceData) -> None:
        problem_id = input_data.problem_id
        exist_keywords = self.problem_dict[problem_id].keywords
        remain_keywords = []
        input_keyword_dict = {input_keyword.id: input_keyword.content for input_keyword in input_data.keywords}
        for exist_keyword in exist_keywords:
            if exist_keyword.id in input_keyword_dict:
                input_keyword_dict.pop(exist_keyword.id)
                remain_keywords.append(exist_keyword)
        is_keyword_changed = len(input_keyword_dict) > 0
        if is_keyword_changed:
            for new_keyword_id, new_keyword_content in input_keyword_dict.items():
                remain_keywords.append(Keyword(id=new_keyword_id, content=new_keyword_content))
            self.problem_dict[problem_id].keywords = remain_keywords
            new_embedded_keywords = self.model.encode([keyword.content for keyword in remain_keywords])
            self.problem_dict[problem_id].embedded_keywords = new_embedded_keywords

    @bentoml.Runnable.method(batchable=False)
    def is_correct_keyword(self, input_data: KeywordInferenceData) -> KeywordInferenceResponse:
        # 키워드 변경 감지도 추가 해야함
        self.synchronize_keywords(input_data)
        if input_data.problem_id not in self.problem_dict:
            self.problem_dict[input_data.problem_id] = Problem(
                subject=None,
                keywords=input_data.keywords,
                embedded_keywords=self.model.encode([keyword.content for keyword in input_data.keywords]),
            )

        problem = self.problem_dict[input_data.problem_id]
        split_answer = input_data.user_answer.strip().split()
        tokenized_answer = []
        for k in range(len(split_answer) - self.word_concat_size + 1):
            tokenized_answer.append(" ".join(split_answer[k : k + self.word_concat_size]))
        if len(split_answer) < self.word_concat_size:
            tokenized_answer.append(" ".join(split_answer))
        tokenized_answer_embedding = self.model.encode(tokenized_answer)
        similarity_scores = cosine_similarity(problem.embedded_keywords, tokenized_answer_embedding)
        predicts = []
        for z, idx in enumerate(similarity_scores.argmax(axis=1)):
            if self.threshold < similarity_scores[z][idx]:
                start_idx = input_data.user_answer.find(tokenized_answer[idx])
                end_idx = start_idx + len(tokenized_answer[idx])
                predicts.append(
                    KeywordResponse(
                        id=problem.keywords[z].id,
                        keyword=problem.keywords[z].content,
                        predict_keyword_position=[start_idx, end_idx],
                        predict_keyword=input_data.user_answer[start_idx:end_idx],
                    )
                )

        return KeywordInferenceResponse(problem_id=input_data.problem_id, correct_keywords=predicts)


keyword_runner = bentoml.Runner(KeywordPredictRunnable, models=[keyword_model])
keyword_service = bentoml.Service(name="keyword_service", runners=[keyword_runner])


@keyword_service.api(
    input=JSON(pydantic_model=KeywordInferenceData),
    output=JSON(pydantic_model=KeywordInferenceResponse),
)
async def keyword_predict(input_data: KeywordInferenceData) -> None:
    result = await keyword_runner.is_correct_keyword.async_run(input_data)
    return result
