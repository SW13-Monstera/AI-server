from typing import Optional

import bentoml
from bentoml.exceptions import NotFound
from bentoml.io import JSON
from sklearn.metrics.pairwise import cosine_similarity

from app.model import save_model
from app.schemas import EmbeddedKeywords, Keyword, KeywordGradingRequest, KeywordGradingResponse, KeywordResponse
from app.utils import ServiceAppFactory  # noqa

try:
    keyword_model = bentoml.pytorch.get("sentence-ko-roberta")
except NotFound:
    save_model()
    keyword_model = bentoml.pytorch.get("sentence-ko-roberta")


class KeywordPredictRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True
    threshold = 0.3
    word_concat_size = 3

    def __init__(self, problem_dict: Optional[dict] = None):
        self.model = bentoml.pytorch.load_model(keyword_model)
        self.problem_dict = problem_dict if problem_dict else {}

    def synchronize_keywords(self, input_data: KeywordGradingRequest) -> None:
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
    def is_correct_keyword(self, input_data: KeywordGradingRequest) -> KeywordGradingResponse:
        if input_data.problem_id not in self.problem_dict:  # 새로운 문제
            self.problem_dict[input_data.problem_id] = EmbeddedKeywords(
                keywords=input_data.keywords,
                embedded_keywords=self.model.encode([keyword.content for keyword in input_data.keywords]),
            )
        else:  # 기존에 있던 문제라면 validation check
            self.synchronize_keywords(input_data)

        problem = self.problem_dict[input_data.problem_id]
        split_answer = input_data.user_answer.strip().split()
        tokenized_answer = []
        for split_answer_start_idx in range(len(split_answer) - self.word_concat_size + 1):
            split_answer_end_idx = split_answer_start_idx + self.word_concat_size
            tokenized_answer.append(" ".join(split_answer[split_answer_start_idx:split_answer_end_idx]))
        if len(split_answer) < self.word_concat_size:
            tokenized_answer.append(" ".join(split_answer))

        tokenized_answer_embedding = self.model.encode(tokenized_answer)
        similarity_scores = cosine_similarity(problem.embedded_keywords, tokenized_answer_embedding)
        predicts = []
        for keyword_idx, embedded_keyword_token_idx in enumerate(similarity_scores.argmax(axis=1)):
            if self.threshold < similarity_scores[keyword_idx][embedded_keyword_token_idx]:
                start_idx = input_data.user_answer.find(tokenized_answer[embedded_keyword_token_idx])
                end_idx = start_idx + len(tokenized_answer[embedded_keyword_token_idx])
                predicts.append(
                    KeywordResponse(
                        id=problem.keywords[keyword_idx].id,
                        keyword=problem.keywords[keyword_idx].content,
                        predict_keyword_position=[start_idx, end_idx],
                        predict_keyword=input_data.user_answer[start_idx:end_idx],
                    )
                )

        return KeywordGradingResponse(problem_id=input_data.problem_id, correct_keywords=predicts)


keyword_runner = bentoml.Runner(KeywordPredictRunnable, models=[keyword_model])
keyword_service = bentoml.Service(name="keyword_service", runners=[keyword_runner])


@keyword_service.api(
    input=JSON(pydantic_model=KeywordGradingRequest),
    output=JSON(pydantic_model=KeywordGradingResponse),
)
async def keyword_predict(input_data: KeywordGradingRequest) -> KeywordGradingResponse:
    """
    {
        "problem_id": 7,
        "user_answer": "쿠키도 만료시간이 있지만 파일로 저장되기 때문에 브라우저를 종료해도 계속해서 정보가 남아 있을 수 있습니다. 또한 만료기간을 넉넉하게 잡아두면 쿠키삭제를 할 때 까지 유지될 수도 있습니다. 반면에 세션도 만료시간을 정할 수 있지만 브라우저가 종료되면 만료시간에 상관없이 삭제됩니다. 예를 들어, 크롬에서 다른 탭을 사용해도 세션을 공유됩니다. 다른 브라우저를 사용하게 되면 다른 세션을 사용할 수 있습니다.",
        "keywords": [
            {
                "id": 0,
                "content":"Lifecycle"
            },
            {
                "id": 1,
                "content":"보안"
            },
            {
                "id": 2,
                "content":"저장위치"
            },
            {
                "id": 3,
                "content":"속도"
            }
        ]
    }
    """  # noqa
    result = await keyword_runner.is_correct_keyword.async_run(input_data)
    return result
