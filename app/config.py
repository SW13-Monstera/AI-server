import logging
import os
import typing
from typing import Literal

import boto3
from dotenv import load_dotenv

load_dotenv()
session = boto3.Session()
secret_manager = session.client(service_name="secretsmanager", region_name="ap-northeast-2")
log = logging.getLogger("__main__")
log.setLevel(logging.INFO)


def get_secret():
    secret_name = f"{STAGE}/cs-broker/ai-server"

    try:
        get_secret_value_response = secret_manager.get_secret_value(SecretId=secret_name)
    except Exception as e:
        log.error("secret manager 에서 config 를 가져 오는 도중 에러가 발생했습니다.")
        raise e
    else:
        secret = eval(get_secret_value_response["SecretString"])
        os.environ.update(secret)


STAGE = typing.cast(Literal["local", "dev", "prod"], os.getenv("STAGE"))

if not STAGE:
    STAGE = "dev"

get_secret()
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
HUGGING_FACE_NAME = "ekzm8523"

CONTENT_MODEL_NAME = "t5"
CONTENT_MODEL_PATH = "google/mt5-base"
KEYWORD_MODEL_PATH = "Huffon/sentence-klue-roberta-base"
KEYWORD_BENTO_MODEL_PATH = f"{STAGE}_keyword_model"
CONTENT_BENTO_MODEL_PATH = f"{STAGE}_content_model"
KEYWORD_MODEL_S3_PATH = os.getenv("KEYWORD_MODEL_S3_PATH")
CONTENT_MODEL_S3_PATH = os.getenv("CONTENT_MODEL_S3_PATH")
