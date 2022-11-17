import logging
import os
from typing import Optional

import torch
from fastapi import Depends
from openprompt import PromptForClassification
from openprompt.plms import get_model_class
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from sentence_transformers import SentenceTransformer

from app.config import (
    CONTENT_MODEL_NAME,
    CONTENT_MODEL_PATH,
    CONTENT_MODEL_S3_PATH,
    KEYWORD_LOCAL_MODEL_PATH,
    KEYWORD_MODEL_PATH,
)
from app.controller.content import ContentController
from app.controller.keyeword import KeywordController
from app.utils.aws_s3 import AwsS3Downloader
from app.utils.utils import get_template_text

log = logging.getLogger("__main__")
s3 = AwsS3Downloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keyword_model: Optional[SentenceTransformer] = None
content_model: Optional[PromptForClassification] = None


def init_model() -> None:
    global keyword_model, content_model

    if os.path.isfile(KEYWORD_LOCAL_MODEL_PATH):
        keyword_model = SentenceTransformer()
        keyword_model.load_state_dict(torch.load(KEYWORD_LOCAL_MODEL_PATH))
    else:
        keyword_model = SentenceTransformer(KEYWORD_MODEL_PATH)
        torch.save(keyword_model.state_dict(), KEYWORD_LOCAL_MODEL_PATH)

    model_class = get_model_class(plm_type=CONTENT_MODEL_NAME)
    plm = model_class.model.from_pretrained(CONTENT_MODEL_PATH)
    tokenizer = model_class.tokenizer.from_pretrained(CONTENT_MODEL_PATH)
    template_text = get_template_text()
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)
    verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=[["yes"], ["no"]])

    content_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)

    model_path = s3.download(url=CONTENT_MODEL_S3_PATH, local_dir=".cache")
    content_model.load_state_dict(torch.load(model_path, map_location=device))


def get_keyword_grading_model() -> SentenceTransformer:
    return keyword_model


def get_content_grading_model() -> PromptForClassification:
    return content_model


def get_content_controller(model: PromptForClassification = Depends(get_content_grading_model)) -> ContentController:
    return ContentController(model)


def get_keyword_controller(model: SentenceTransformer = Depends(get_keyword_grading_model)) -> KeywordController:
    return KeywordController(model)


init_model()
