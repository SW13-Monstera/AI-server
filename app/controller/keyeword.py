import logging
import re
from pprint import pformat
from typing import List, Optional, Tuple

import torch.cuda
from konlpy.tag import Kkma
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.controller.base import BaseController
from app.decorator import singleton
from app.schemas import (
    KeywordGradingRequest,
    KeywordGradingResponse,
    KeywordResponse,
    KeywordSimilarityInfo,
    KeywordStandard,
    Problem,
)
from app.utils.utils import get_stopwords

log = logging.getLogger("__main__")


@singleton
class KeywordController(BaseController):
    def __init__(self, model: SentenceTransformer, problem_dict: Optional[dict] = None):
        self.problem_dict = problem_dict if problem_dict else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"keyword predict model is running on {self.device}")
        self.model = model.to(self.device)
        self.tokenizer = Kkma()
        self.stopwords = get_stopwords()
        self.threshold = 0.7
        self.word_concat_size = 2

    def create_problem(self, input_data: KeywordGradingRequest) -> None:
        log.info(f"problem id [{input_data.problem_id}] : create problem")
        keyword_standards: List[KeywordStandard] = input_data.keyword_standards

        self.problem_dict[input_data.problem_id] = Problem(
            keyword_standards=keyword_standards,
            embedded_keywords=self.model.encode([standard.content for standard in keyword_standards]),
        )

    def synchronize_keywords(self, input_data: KeywordGradingRequest) -> None:
        problem_id = input_data.problem_id
        if self._is_new_problem(problem_id):  # 새로운 문제
            self.create_problem(input_data)
            return

        pre_keyword_id_set = set(keyword.id for keyword in self.problem_dict[problem_id].keyword_standards)
        new_keyword_id_set = set(keyword.id for keyword in input_data.keyword_standards)
        if pre_keyword_id_set != new_keyword_id_set:
            self.problem_dict.pop(problem_id)
            self.create_problem(input_data)

    def _is_new_problem(self, problem_id: int):
        return problem_id not in self.problem_dict

    def get_tokenized_answer(self, user_answer: str) -> List[str]:
        regex_filter = r"[^\uAC00-\uD7A3a-zA-Z\s]"
        user_answer = re.sub(regex_filter, "", user_answer)
        tokenized_answers = tuple(
            word for word, _ in self.tokenizer.pos(user_answer) if word not in self.stopwords and len(word) > 1
        )
        tokenized_words = []
        for concat_size in range(1, self.word_concat_size + 1):
            for split_answer_start_idx in range(len(tokenized_answers) - concat_size + 1):
                split_answer_end_idx = split_answer_start_idx + concat_size
                tokenized_words.append(" ".join(tokenized_answers[split_answer_start_idx:split_answer_end_idx]))
        log.info(tokenized_words)
        return tokenized_words

    def calculate_keyword_score(
        self,
        keyword_standards: List[KeywordStandard],
        split_keyword_standards: List[KeywordStandard],
        similarity_scores: NDArray,
    ) -> Tuple[KeywordSimilarityInfo]:
        keyword_similarity_info_dict = {
            standard.id: KeywordSimilarityInfo(
                id=standard.id, content=standard.content, score=0, tokenized_answer_idx=-1
            )
            for standard in keyword_standards
        }

        for keyword_idx, tokenized_answer_idx in enumerate(similarity_scores.argmax(axis=1)):
            target_keyword_id = split_keyword_standards[keyword_idx].id
            score = similarity_scores[keyword_idx][tokenized_answer_idx]
            keyword_similarity_info = keyword_similarity_info_dict[target_keyword_id]
            if self.threshold < score and keyword_similarity_info.score < score:
                keyword_similarity_info.tokenized_answer_idx = tokenized_answer_idx
                keyword_similarity_info.score = score

        return tuple(keyword_similarity_info_dict.values())

    @staticmethod
    def get_predicted_keyword_position(
        tokenized_answer: List[str], embedded_keyword_token_idx: int, user_answer: str
    ) -> Tuple[int, int]:
        split_word = tokenized_answer[embedded_keyword_token_idx].split()
        first_word, last_word = split_word[0], split_word[-1]
        start_idx = user_answer.find(first_word)
        end_idx = user_answer.find(last_word) + len(last_word)
        return start_idx, end_idx

    async def grading(self, input_data: KeywordGradingRequest) -> KeywordGradingResponse:
        log.info(pformat(input_data.__dict__))
        self.synchronize_keywords(input_data)

        problem = self.problem_dict[input_data.problem_id]
        tokenized_answer = self.get_tokenized_answer(input_data.user_answer)
        predicts = []
        if tokenized_answer:
            tokenized_answer_embedding = self.model.encode(tokenized_answer)
            similarity_scores = cosine_similarity(problem.embedded_keywords, tokenized_answer_embedding)
            keyword_similarity_infos = self.calculate_keyword_score(
                keyword_standards=input_data.keyword_standards,
                split_keyword_standards=problem.keyword_standards,
                similarity_scores=similarity_scores,
            )
            for keyword_similarity_info in keyword_similarity_infos:
                if keyword_similarity_info.score >= self.threshold:
                    start_idx, end_idx = self.get_predicted_keyword_position(
                        tokenized_answer, keyword_similarity_info.tokenized_answer_idx, input_data.user_answer
                    )
                    predicts.append(
                        KeywordResponse(
                            id=keyword_similarity_info.id,
                            keyword=keyword_similarity_info.content,
                            predict_keyword_position=[start_idx, end_idx],
                            predict_keyword=input_data.user_answer[start_idx:end_idx],
                        )
                    )
        response_data = KeywordGradingResponse(problem_id=input_data.problem_id, correct_keywords=predicts)
        log.info(pformat(response_data.__dict__))
        return response_data
