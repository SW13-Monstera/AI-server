import random

import numpy as np
import pandas
import pandas as pd
import pytest
from openprompt import PromptForClassification
from sentence_transformers import SentenceTransformer

from app.model import get_content_grading_model, get_keyword_grading_model
from app.runnable import KeywordPredictRunnable
from app.schemas import ContentGradingRequest, ContentStandard, KeywordGradingRequest, KeywordStandard, Problem


@pytest.fixture(scope="session")
def keyword_model() -> SentenceTransformer:
    return get_keyword_grading_model()


@pytest.fixture(scope="session")
def content_model() -> PromptForClassification:
    return get_content_grading_model()


@pytest.fixture(scope="session")
def user_answer_df(
    path: str = "/Users/minjaewon/workspace/AI-server/app/static/changed_user_answer.csv",
) -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def problem_dict(keyword_model: SentenceTransformer, user_answer_df) -> dict:
    problem_dict = {}
    keyword_id = 0
    # criterion parsing
    for _, data in user_answer_df.iterrows():
        problem_id = data["problem_id"]
        if problem_id not in problem_dict:
            keyword_standards = []
            for content in eval(data["keyword_criterion"]):
                if content is np.nan:
                    content = "NULL"
                if "," in content:
                    for split_content in content.split(","):
                        keyword_standards.append(KeywordStandard(id=keyword_id, content=split_content))
                else:
                    keyword_standards.append(KeywordStandard(id=keyword_id, content=content))
                keyword_id += 1

            embedded_keywords = keyword_model.encode([keyword.content for keyword in keyword_standards])
            problem_dict[problem_id] = Problem(
                keyword_standards=keyword_standards,
                embedded_keywords=embedded_keywords,
            )
    return problem_dict


@pytest.fixture(scope="function")
def random_keyword_data(problem_dict: dict, user_answer_df) -> KeywordGradingRequest:
    random_idx = random.randint(0, len(user_answer_df) - 1)
    random_data = user_answer_df.iloc[random_idx]
    problem_id = random_data["problem_id"]
    keyword_standards = problem_dict[problem_id].keyword_standards
    return KeywordGradingRequest(
        problem_id=problem_id, user_answer=random_data.user_answer, keyword_standards=keyword_standards
    )


@pytest.fixture(scope="function")
def random_multi_candidate_keyword_data(problem_dict: dict, user_answer_df) -> KeywordGradingRequest:
    flag = True
    while flag:
        random_idx = random.randint(0, len(user_answer_df) - 1)
        random_data = user_answer_df.iloc[random_idx]
        standard_contents = [standard.content for standard in problem_dict[random_data.problem_id].keyword_standards]
        for criterion in eval(random_data.keyword_criterion):
            err_message = f"static data file에 무결성이 깨졌습니다. problem : {random_data.problem}"
            if "," in criterion:
                for split_criterion in criterion.split(", "):
                    assert split_criterion in standard_contents, err_message
                flag = False
                break
            assert criterion in standard_contents, err_message
    keyword_standards = []

    for i, keyword_content in enumerate(eval(random_data.keyword_criterion)):
        keyword_standards.append(KeywordStandard(id=random_idx * 10 + i + 1, content=keyword_content))
    return KeywordGradingRequest(
        problem_id=random_data.problem_id, user_answer=random_data.user_answer, keyword_standards=keyword_standards
    )


@pytest.fixture(scope="function")
def random_content_data(user_answer_df) -> ContentGradingRequest:
    random_idx = random.randint(0, len(user_answer_df) - 1)
    random_series = user_answer_df.iloc[random_idx]
    content_standards = []
    offset = random.randint(0, 10000)
    for i, content in enumerate(eval(random_series["scoring_criterion"])):
        content_standards.append(ContentStandard(id=offset + i, content=content))

    return ContentGradingRequest(
        problem_id=random_idx, user_answer=random_series.user_answer, content_standards=content_standards
    )


@pytest.fixture(scope="session")
def keyword_runnable() -> KeywordPredictRunnable:
    return KeywordPredictRunnable()
