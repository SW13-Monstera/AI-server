from typing import Optional

import bentoml
import torch.cuda
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


class KeywordPredictRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True
    threshold = 0.5
    word_concat_size = 2

    def __init__(self, problem_dict: Optional[dict] = None):
        self.model = get_keyword_grading_model()
        self.problem_dict = problem_dict if problem_dict else {}

    def synchronize_keywords(self, input_data: KeywordGradingRequest) -> None:
        problem_id = input_data.problem_id
        exist_keywords = self.problem_dict[problem_id].keyword_standards
        remain_keywords = []
        input_keyword_dict = {input_keyword.id: input_keyword.content for input_keyword in input_data.keyword_standards}
        for exist_keyword in exist_keywords:
            if exist_keyword.id in input_keyword_dict:
                input_keyword_dict.pop(exist_keyword.id)
                remain_keywords.append(exist_keyword)
        is_keyword_changed = len(input_keyword_dict) > 0
        if is_keyword_changed:
            for new_keyword_id, new_keyword_content in input_keyword_dict.items():
                remain_keywords.append(KeywordStandard(id=new_keyword_id, content=new_keyword_content))
            self.problem_dict[problem_id].keyword_standards = remain_keywords
            new_embedded_keywords = self.model.encode([keyword.content for keyword in remain_keywords])
            self.problem_dict[problem_id].embedded_keywords = new_embedded_keywords

    @bentoml.Runnable.method(batchable=False)
    def is_correct_keyword(self, input_data: KeywordGradingRequest) -> KeywordGradingResponse:
        if input_data.problem_id not in self.problem_dict:  # 새로운 문제
            self.problem_dict[input_data.problem_id] = Problem(
                keyword_standards=input_data.keyword_standards,
                embedded_keywords=self.model.encode([keyword.content for keyword in input_data.keyword_standards]),
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
                        id=problem.keyword_standards[keyword_idx].id,
                        keyword=problem.keyword_standards[keyword_idx].content,
                        predict_keyword_position=[start_idx, end_idx],
                        predict_keyword=input_data.user_answer[start_idx:end_idx],
                    )
                )

        return KeywordGradingResponse(problem_id=input_data.problem_id, correct_keywords=predicts)


class ContentPredictRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.model = get_content_grading_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.template = self.model.template
        self.verbalizer = self.model.verbalizer
        special_tokens_dict = {"additional_special_tokens": ["</s>", "<unk>", "<pad>"]}
        self.model.tokenizer.add_special_tokens(special_tokens_dict)
        self.wrapped_tokenizer = T5TokenizerWrapper(
            max_seq_length=256, decoder_max_length=3, tokenizer=self.model.tokenizer, truncate_method="head"
        )

    @bentoml.Runnable.method(batchable=False)
    def is_correct_content(self, input_data: ContentGradingRequest) -> ContentGradingResponse:
        user_answer = self._preprocessing(input_data.user_answer).strip()
        input_data_list = []
        for i, content_standard in enumerate(input_data.content_standards):
            input_data_list.append(InputExample(text_a=user_answer, text_b=content_standard.content.strip(), guid=i))

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
                guids = model_inputs.guid.cpu().numpy()
                for guid, predict in zip(guids, predicts):
                    if predict == 1:
                        correct_contents.append(
                            ContentResponse(id=guid, content=input_data.content_standards[guid].content)
                        )
                del model_inputs, logits
        torch.cuda.empty_cache()

        return ContentGradingResponse(problem_id=input_data.problem_id, correct_contents=correct_contents)
