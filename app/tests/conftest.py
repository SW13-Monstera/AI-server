import random

import pandas
import pandas as pd
import pytest
from openprompt import PromptForClassification
from sentence_transformers import SentenceTransformer

from app.model import get_content_grading_model, get_keyword_grading_model
from app.schemas import ContentGradingRequest, ContentStandard, KeywordGradingRequest, KeywordStandard, Problem


@pytest.fixture(scope="session")
def keyword_model() -> SentenceTransformer:
    return get_keyword_grading_model()


@pytest.fixture(scope="session")
def content_model() -> PromptForClassification:
    return get_content_grading_model()


@pytest.fixture(scope="session")
def problem_df(path: str = "app/static/user_answer.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def problem_dict(keyword_model: SentenceTransformer, problem_df: pd.DataFrame) -> dict:
    problem_dict = {}
    keyword_id = 0
    # criterion parsing
    for _, data in problem_df.iterrows():
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
def random_keyword_data(problem_dict: dict, problem_df: pd.DataFrame) -> KeywordGradingRequest:
    random_idx = random.randint(0, len(problem_df) - 1)
    random_data = problem_df.iloc[random_idx]
    problem_id = random_data["problem_id"]
    keyword_standards = problem_dict[problem_id].keyword_standards
    return KeywordGradingRequest(
        problem_id=problem_id, user_answer=random_data.user_answer, keyword_standards=keyword_standards
    )


@pytest.fixture(scope="function")
def random_content_data(problem_df: pd.DataFrame) -> ContentGradingRequest:
    random_idx = random.randint(0, len(problem_df) - 1)
    random_series = problem_df.iloc[random_idx]
    content_standards = []
    offset = random.randint(0, 10000)
    for i, criterion in enumerate(eval(random_series["scoring_criterion"])):
        content, _ = map(str.strip, criterion.split("-"))
        content_standards.append(ContentStandard(id=offset + i, content=content))

    return ContentGradingRequest(
        problem_id=random_idx, user_answer=random_series.user_answer, content_standards=content_standards
    )
