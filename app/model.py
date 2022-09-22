import logging

import bentoml
import torch
from bentoml.exceptions import NotFound
from openprompt import PromptForClassification
from openprompt.plms import get_model_class
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from sentence_transformers import SentenceTransformer

from app.config import (
    CONTENT_BENTO_MODEL_PATH,
    CONTENT_MODEL_NAME,
    CONTENT_MODEL_PATH,
    CONTENT_MODEL_S3_PATH,
    KEYWORD_BENTO_MODEL_PATH,
    KEYWORD_MODEL_PATH,
)
from app.utils.aws_s3 import AwsS3Downloader

log = logging.getLogger("__main__")
s3 = AwsS3Downloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_keyword_grading_model() -> SentenceTransformer:
    try:
        model = bentoml.pytorch.load_model(KEYWORD_BENTO_MODEL_PATH)
    except NotFound:
        model = SentenceTransformer(KEYWORD_MODEL_PATH)
        bentoml.pytorch.save_model(KEYWORD_BENTO_MODEL_PATH, model)
    return model


def get_content_grading_model() -> PromptForClassification:
    try:
        model = bentoml.pytorch.load_model(CONTENT_BENTO_MODEL_PATH)

    except NotFound:
        model_class = get_model_class(plm_type=CONTENT_MODEL_NAME)
        plm = model_class.model.from_pretrained(CONTENT_MODEL_PATH)
        tokenizer = model_class.tokenizer.from_pretrained(CONTENT_MODEL_PATH)
        template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
        template = ManualTemplate(tokenizer=tokenizer, text=template_text)
        verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=[["yes"], ["no"]])
        model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)

        model_path = s3.download(url=CONTENT_MODEL_S3_PATH, local_dir=".cache")
        model.load_state_dict(torch.load(model_path, map_location=device))
        bentoml.pytorch.save_model(CONTENT_BENTO_MODEL_PATH, model)
    return model
