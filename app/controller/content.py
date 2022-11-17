import logging
from pprint import pformat

import torch.cuda
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import T5TokenizerWrapper

from app.schemas import ContentGradingRequest, ContentGradingResponse, ContentResponse

log = logging.getLogger("__main__")


class ContentController:
    def __init__(self, model: PromptForClassification):
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

    async def is_correct_content(self, input_data: ContentGradingRequest) -> ContentGradingResponse:
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
