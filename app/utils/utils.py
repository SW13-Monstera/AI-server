# flake8: noqa
from pathlib import Path
from typing import Set

from app.config import STOPWORD_FILE_PATH


def get_stopwords() -> Set:
    return set(Path(STOPWORD_FILE_PATH).read_text(encoding="UTF-8").split("\n"))


def get_template_text() -> str:
    return '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'


def get_keyword_grading_request_example() -> dict:
    return {
        "problem_id": 1234,
        "user_answer": "먼저 저장위치에 차이가 있습니다. 쿠키는 클라이언트측에 저장되고 세션은 서버측에 저장이 됩니다. 세션은 서버를 거쳐야 하기 때문에 세션이 쿠키보다 속도가 느린 대신 보안에 유리합니다.그리고 라이프사이클에 차이가 있는데 둘 다 만료기간을 정해놓지만 세션은 브라우저가 종료되면 세션 스토리지에 세션 아이디가 사라지기 때문에 만료기간에 상관없이 삭제됩니다.",  # noqa
        "keyword_standards": [
            {"id": 123, "content": "저장위치"},
            {"id": 124, "content": "보안"},
            {"id": 125, "content": "라이프사이클"},
            {"id": 126, "content": "속도"},
        ],
    }


def get_integrated_grading_request_example() -> dict:
    return {
        "problem_id": 1234,
        "user_answer": "먼저 저장위치에 차이가 있습니다. 쿠키는 클라이언트측에 저장되고 세션은 서버측에 저장이 됩니다. 세션은 서버를 거쳐야 하기 때문에 세션이 쿠키보다 속도가 느린 대신 보안에 유리합니다.그리고 라이프사이클에 차이가 있는데 둘 다 만료기간을 정해놓지만 세션은 브라우저가 종료되면 세션 스토리지에 세션 아이디가 사라지기 때문에 만료기간에 상관없이 삭제됩니다.",
        # noqa
        "keyword_standards": [
            {"id": 123, "content": "저장위치"},
            {"id": 124, "content": "보안"},
            {"id": 125, "content": "라이프사이클"},
            {"id": 126, "content": "속도"},
        ],
        "content_standards": [
            {"id": 127, "content": "쿠키는 클라이언트측에 저장되고, 세션은 서버에 저장이 된다."},
            {"id": 128, "content": "세션이 쿠키보다 보안에 유리하다."},
            {"id": 129, "content": "세션이 쿠키보다 속도가 느리다."},
            {"id": 130, "content": "세션은 브라우저가 종료되면 만료기간이 남아있어도 삭제된다."},
        ],
    }


def get_keyword_grading_response_example() -> dict:
    return {
        "problem_id": 1234,
        "correct_keywords": [
            {"id": 123, "keyword": "저장위치", "predict_keyword_position": [3, 7], "predict_keyword": "저장위치"},
            {"id": 124, "keyword": "보안", "predict_keyword_position": [92, 94], "predict_keyword": "보안"},
            {"id": 125, "keyword": "라이프사이클", "predict_keyword_position": [106, 112], "predict_keyword": "라이프사이클"},
            {"id": 126, "keyword": "속도", "predict_keyword_position": [82, 84], "predict_keyword": "속도"},
        ],
    }


def get_integrated_grading_response_example() -> dict:
    return {
        "problem_id": 1234,
        "correct_contents": [
            {"id": 127, "content": "쿠키는 클라이언트측에 저장되고, 세션은 서버에 저장이 된다."},
            {"id": 128, "content": "세션이 쿠키보다 보안에 유리하다."},
            {"id": 129, "content": "세션이 쿠키보다 속도가 느리다."},
            {"id": 130, "content": "세션은 브라우저가 종료되면 만료기간이 남아있어도 삭제된다."},
        ],
        "correct_keywords": [
            {"id": 123, "keyword": "저장위치", "predict_keyword_position": [3, 7], "predict_keyword": "저장위치"},
            {"id": 124, "keyword": "보안", "predict_keyword_position": [92, 94], "predict_keyword": "보안"},
            {"id": 125, "keyword": "라이프사이클", "predict_keyword_position": [106, 112], "predict_keyword": "라이프사이클"},
            {"id": 126, "keyword": "속도", "predict_keyword_position": [82, 84], "predict_keyword": "속도"},
        ],
    }
