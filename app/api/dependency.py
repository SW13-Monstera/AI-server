import logging
import os
from typing import Optional

import torch
from fastapi import Depends
from openprompt import PromptForClassification
from openprompt.plms import get_model_class
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from sentence_transformers import SentenceTransformer
from torch.types import Device

from app.config import (
    CONTENT_LOCAL_MODEL_PATH,
    CONTENT_MODEL_NAME,
    CONTENT_MODEL_PATH,
    CONTENT_MODEL_S3_PATH,
    KEYWORD_LOCAL_MODEL_PATH,
    KEYWORD_MODEL_PATH,
)
from app.controller.content import ContentController
from app.controller.keyeword import KeywordController
from app.decorator import singleton
from app.utils.aws_s3 import AwsS3Downloader
from app.utils.utils import get_template_text

log = logging.getLogger("__main__")


@singleton
class ApplicationContext:
    _s3: AwsS3Downloader
    _device: Device
    _keyword_model: Optional[SentenceTransformer] = None
    _content_model: Optional[PromptForClassification] = None
    _keyword_controller: Optional[KeywordController] = None
    _content_controller: Optional[ContentController] = None

    def __init__(self):
        self._s3 = AwsS3Downloader()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._keyword_model = self._load_or_pull_keyword_model()
        self._keyword_model.eval()
        self._content_model = self._load_or_pull_content_model()
        self._content_model.eval()
        self._keyword_controller = KeywordController(self._keyword_model)
        self._content_controller = ContentController(self._content_model)

    def _load_or_pull_keyword_model(self):
        if os.path.exists(KEYWORD_LOCAL_MODEL_PATH):
            return torch.load(KEYWORD_LOCAL_MODEL_PATH)
        keyword_model = SentenceTransformer(KEYWORD_MODEL_PATH)
        torch.save(keyword_model, KEYWORD_LOCAL_MODEL_PATH)
        return keyword_model

    def _load_or_pull_content_model(self):
        if os.path.exists(CONTENT_LOCAL_MODEL_PATH):
            return torch.load(CONTENT_LOCAL_MODEL_PATH)
        return self.pull_content_model_from_s3()

    def pull_content_model_from_s3(self):
        model_class = get_model_class(plm_type=CONTENT_MODEL_NAME)
        plm = model_class.model.from_pretrained(CONTENT_MODEL_PATH)
        tokenizer = model_class.tokenizer.from_pretrained(CONTENT_MODEL_PATH)
        template_text = get_template_text()

        template = ManualTemplate(tokenizer=tokenizer, text=template_text)
        verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=[["yes"], ["no"]])
        content_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)
        model_path = self._s3.download(url=CONTENT_MODEL_S3_PATH, local_dir=".cache")
        content_model.load_state_dict(torch.load(model_path, map_location=self._device))

        torch.save(content_model, CONTENT_LOCAL_MODEL_PATH)
        return content_model

    def get_keyword_model(self) -> SentenceTransformer:
        return self._keyword_model

    def get_content_model(self) -> PromptForClassification:
        return self._content_model

    def get_keyword_controller(self) -> KeywordController:
        return self._keyword_controller

    def get_content_controller(self) -> ContentController:
        return self._content_controller

    def get_device(self) -> Device:
        return self._device


def load_application_context(context: ApplicationContext = Depends()):
    ...
