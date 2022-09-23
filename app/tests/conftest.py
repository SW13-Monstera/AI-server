import random

import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

from app.model import get_keyword_grading_model
from app.schemas import KeywordGradingRequest, KeywordStandard, Problem


@pytest.fixture(scope="session")
def init_save_model() -> None:
    get_keyword_grading_model()


@pytest.fixture(scope="session")
def keyword_model(init_save_model) -> SentenceTransformer:
    return get_keyword_grading_model()


@pytest.fixture(scope="session")
def problem_dict(keyword_model, path: str = "app/static/user_answer.csv") -> dict:
    df = pd.read_csv(path)
    problem_dict = {}
    keyword_id = 0
    # criterion parsing
    for _, data in df.iterrows():
        problem_id = data["problem_id"]
        if problem_id not in problem_dict:
            keyword_standards = []

            for criterion in eval(data["keyword_criterion"]):
                content, _ = map(str.strip, criterion.split("-"))
                keyword_standards.append(KeywordStandard(id=keyword_id, content=content))
                keyword_id += 1

            embedded_keywords = keyword_model.encode([keyword.content for keyword in keyword_standards])
            problem_dict[problem_id] = Problem(
                keyword_standards=keyword_standards,
                embedded_keywords=embedded_keywords,
            )
    return problem_dict


@pytest.fixture(scope="function")
def random_keyword_data(problem_dict: dict, path: str = "app/static/user_answer.csv") -> KeywordGradingRequest:
    df = pd.read_csv(path)
    random_idx = random.randint(0, len(df) - 1)
    random_data = df.iloc[random_idx]
    problem_id = random_data["problem_id"]
    keyword_standards = problem_dict[problem_id].keyword_standards
    return KeywordGradingRequest(
        problem_id=problem_id, user_answer=random_data.user_answer, keyword_standards=keyword_standards
    )
