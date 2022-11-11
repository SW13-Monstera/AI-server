import logging
import re
from pprint import pformat
from typing import List, Optional, Tuple

import bentoml
import torch.cuda
from konlpy.tag import Mecab
from numpy.typing import NDArray
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms import T5TokenizerWrapper
from sklearn.metrics.pairwise import cosine_similarity

from app.config import MECAB_DIC_PATH, OS
from app.model import get_content_grading_model, get_keyword_grading_model
from app.schemas import (
    ContentGradingRequest,
    ContentGradingResponse,
    ContentResponse,
    KeywordGradingRequest,
    KeywordGradingResponse,
    KeywordResponse,
    KeywordSimilarityInfo,
    KeywordStandard,
    Problem,
)
from app.utils.utils import get_stopwords

log = logging.getLogger("__main__")


if OS == "Windows":
    import win32file

    win32file._setmaxstdio(2048)


class KeywordPredictRunnable(bentoml.Runnable):
    SUPPORTS_CPU_MULTI_THREADING = True
    threshold = 0.7
    word_concat_size = 2

    def __init__(self, problem_dict: Optional[dict] = None):
        self.problem_dict = problem_dict if problem_dict else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"keyword predict model is running on {self.device}")
        self.model = get_keyword_grading_model().to(self.device)
        self.tokenizer = Mecab(MECAB_DIC_PATH) if OS == "Windows" else Mecab()
        self.stopwords = get_stopwords()

    def create_problem(self, input_data: KeywordGradingRequest) -> None:
        log.info(f"problem id [{input_data.problem_id}] : create problem")
        keyword_standards = []
        for keyword_standard in input_data.keyword_standards:
            for content in keyword_standard.content.split(", "):
                keyword_standards.append(KeywordStandard(id=keyword_standard.id, content=content.strip()))

        self.problem_dict[input_data.problem_id] = Problem(
            keyword_standards=keyword_standards,
            embedded_keywords=self.model.encode([standard.content for standard in keyword_standards]),
        )

    def synchronize_keywords(self, input_data: KeywordGradingRequest) -> None:
        problem_id = input_data.problem_id
        if problem_id not in self.problem_dict:  # 새로운 문제
            self.create_problem(input_data)
        else:  # 기존에 존재하던 문제
            pre_keyword_id_set = set(keyword.id for keyword in self.problem_dict[problem_id].keyword_standards)
            new_keyword_id_set = set(keyword.id for keyword in input_data.keyword_standards)
            if pre_keyword_id_set != new_keyword_id_set:
                self.problem_dict.pop(problem_id)
                self.create_problem(input_data)

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

    @bentoml.Runnable.method(batchable=False)
    def is_correct_keyword(self, input_data: KeywordGradingRequest) -> KeywordGradingResponse:
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


class ContentPredictRunnable(bentoml.Runnable):

    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        model = get_content_grading_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"content predict model is running on : {self.device}")
        self.template = model.template
        self.verbalizer = model.verbalizer
        special_tokens_dict = {"additional_special_tokens": ["</s>", "<unk>", "<pad>"]}
        model.tokenizer.add_special_tokens(special_tokens_dict)
        self.wrapped_tokenizer = T5TokenizerWrapper(
            max_seq_length=256, decoder_max_length=3, tokenizer=model.tokenizer, truncate_method="head"
        )
        self.model = model.to(self.device)

    @staticmethod
    def is_correct(predict) -> bool:
        return predict == 1

    @bentoml.Runnable.method(batchable=False)
    def is_correct_content(self, input_data: ContentGradingRequest) -> ContentGradingResponse:
        log.info(pformat(input_data.__dict__))
        user_answer = input_data.user_answer.strip()
        input_data_list = [
            InputExample(text_a=user_answer, text_b=content_standard.content.strip(), guid=content_standard.id)
            for content_standard in input_data.content_standards
        ]

        data_loader = PromptDataLoader(
            dataset=input_data_list,
            template=self.template,
            tokenizer=self.model.tokenizer,
            tokenizer_wrapper_class=T5TokenizerWrapper,
            max_seq_length=256,
            decoder_max_length=3,
            predict_eos_token=False,
            truncate_method="head",
            batch_size=len(input_data_list),
        )
        correct_contents = []
        with torch.no_grad():
            for model_inputs in data_loader:
                model_inputs = model_inputs.to(self.device)
                logits = self.model(model_inputs)
                predicts = torch.argmax(logits, dim=1).cpu().numpy()
                correct_contents.extend(
                    ContentResponse(id=input_data_list[idx].guid, content=input_data_list[idx].text_b)
                    for idx, predict in enumerate(predicts)
                    if self.is_correct(predict)
                )

                del model_inputs, logits
        torch.cuda.empty_cache()
        response_data = ContentGradingResponse(problem_id=input_data.problem_id, correct_contents=correct_contents)
        log.info(pformat(response_data.__dict__))
        return response_data
