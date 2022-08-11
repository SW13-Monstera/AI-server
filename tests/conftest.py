import random
from typing import Generator

import bentoml
import pandas as pd
import pytest
from bentoml import Runner, Service
from bentoml.models import Model
from bentoml.testing.server import host_bento

from schemas import Keyword, KeywordInferenceData, Problem
from service import KeywordPredictRunnable


@pytest.fixture(scope="session")
def keyword_model() -> Model:
    return bentoml.pytorch.get("sentence-ko-roberta")


@pytest.fixture(scope="session")
def keyword_runner(keyword_model) -> Runner:
    return bentoml.Runner(KeywordPredictRunnable, models=[keyword_model])


@pytest.fixture(scope="session")
def keyword_service(keyword_runner) -> Service:
    return bentoml.Service(name="keyword_service", runners=[keyword_runner])


@pytest.fixture(scope="session")
def problem_dict(keyword_model, path: str = "user_answer.csv") -> dict:
    pytorch_keyword_model = bentoml.pytorch.load_model(keyword_model)
    df = pd.read_csv(path)
    problem_dict = {}
    keyword_id = 0
    # criterion parsing
    for i, data in df.iterrows():
        problem_id = data["problem_id"]
        if problem_id not in problem_dict:
            keywords = []

            for criterion in eval(data["keyword_criterion"]):
                content, _ = map(str.strip, criterion.split("-"))
                keywords.append(Keyword(id=keyword_id, content=content))
                keyword_id += 1

            embedded_keywords = pytorch_keyword_model.encode(
                [keyword.content for keyword in keywords]
            )
            problem_dict[problem_id] = Problem(
                subject=data["problem"],
                keywords=keywords,
                embedded_keywords=embedded_keywords,
            )
    return problem_dict


@pytest.fixture(scope="function")
def random_keyword_data(
    problem_dict: dict, path: str = "user_answer.csv"
) -> KeywordInferenceData:
    df = pd.read_csv(path)
    random_idx = random.randint(0, len(df) - 1)
    random_data = df.iloc[random_idx]
    problem_id = random_data["problem_id"]
    keywords = problem_dict[problem_id].keywords
    return KeywordInferenceData(
        problem_id=problem_id, user_answer=random_data.user_answer, keywords=keywords
    )


@pytest.fixture(scope="session")
def host() -> Generator[str, None, None]:

    with host_bento(bento="keyword_service") as host:
        yield host
