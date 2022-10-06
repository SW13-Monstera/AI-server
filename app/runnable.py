import logging
import re
from pprint import pformat
from typing import List, Optional

import bentoml
import torch.cuda
from konlpy.tag import Mecab
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms import T5TokenizerWrapper
from sklearn.metrics.pairwise import cosine_similarity

from app.model import get_content_grading_model, get_keyword_grading_model
from app.schemas import (
    ContentGradingRequest,
    ContentGradingResponse,
    ContentResponse,
    KeywordGradingRequest,
    KeywordGradingResponse,
    KeywordResponse,
    KeywordStandard,
    Problem,
)
from app.utils.utils import get_stopwords

log = logging.getLogger("__main__")


class KeywordPredictRunnable(bentoml.Runnable):
    SUPPORTS_CPU_MULTI_THREADING = True
    threshold = 0.5
    word_concat_size = 2

    def __init__(self, problem_dict: Optional[dict] = None):
        self.problem_dict = problem_dict if problem_dict else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"keyword predict model is running on {self.device}")
        self.model = get_keyword_grading_model().to(self.device)
        self.tokenizer = Mecab()
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

    @bentoml.Runnable.method(batchable=False)
    def is_correct_keyword(self, input_data: KeywordGradingRequest) -> KeywordGradingResponse:
        log.info(pformat(input_data.__dict__))
        self.synchronize_keywords(input_data)

        problem = self.problem_dict[input_data.problem_id]
        tokenized_answer = self.get_tokenized_answer(input_data.user_answer)
        tokenized_answer_embedding = self.model.encode(tokenized_answer)
        similarity_scores = cosine_similarity(problem.embedded_keywords, tokenized_answer_embedding)

        predicts = []
        keyword_score_dict = {standard.id: [None, 0, standard.content] for standard in input_data.keyword_standards}
        for keyword_idx, embedded_keyword_token_idx in enumerate(similarity_scores.argmax(axis=1)):
            target_keyword_id = problem.keyword_standards[keyword_idx].id
            score = similarity_scores[keyword_idx][embedded_keyword_token_idx]
            if self.threshold < score and keyword_score_dict[target_keyword_id][1] < score:
                keyword_score_dict[target_keyword_id][:2] = embedded_keyword_token_idx, score

        for keyword_id, (embedded_keyword_token_idx, score, content) in keyword_score_dict.items():
            if score > self.threshold:
                split_word = tokenized_answer[embedded_keyword_token_idx].split()
                first_word, last_word = split_word[0], split_word[-1]
                start_idx = input_data.user_answer.find(first_word)
                end_idx = input_data.user_answer.find(last_word) + len(last_word)
                predicts.append(
                    KeywordResponse(
                        id=keyword_id,
                        keyword=content,
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
