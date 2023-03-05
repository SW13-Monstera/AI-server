import logging
import os
import platform
import typing
from typing import Literal

import boto3
import pyrootutils
from dotenv import load_dotenv

load_dotenv()
session = boto3.Session()
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
secret_manager = session.client(
    service_name="secretsmanager",
    region_name="ap-northeast-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

log = logging.getLogger("__main__")
log.setLevel(logging.INFO)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def get_secret():
    secret_name = "csbroker"

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
KEYWORD_LOCAL_MODEL_PATH = os.path.join(root, f"app/static/{STAGE}_keyword_model")
CONTENT_LOCAL_MODEL_PATH = os.path.join(root, f"app/static/{STAGE}_content_model")
CONTENT_MODEL_S3_PATH = os.getenv("CONTENT_MODEL_S3_PATH")
STOPWORD_FILE_PATH = os.path.join(root, "app/static/stopwords.txt")
OS = platform.system()
MECAB_DIC_PATH = "C:\mecab/mecab-ko-dic"  # noqa
PROJECT_NAME = "csbroker-ai"
API_V1_STR: str = "/api/v1"
TEST_AI_SERVER_HOST: str = os.getenv("TEST_AI_SERVER_HOST")
