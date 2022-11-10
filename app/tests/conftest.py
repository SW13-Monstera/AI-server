import pandas as pd
import pytest
from openprompt import PromptForClassification
from sentence_transformers import SentenceTransformer

from app.model import get_content_grading_model, get_keyword_grading_model
from app.runnable import KeywordPredictRunnable
from app.schemas import ContentGradingRequest, KeywordGradingRequest
from app.tests.factory import ContentDataFactory, KeywordDataFactory


@pytest.fixture(scope="session")
def keyword_model() -> SentenceTransformer:
    return get_keyword_grading_model()


@pytest.fixture(scope="session")
def content_model() -> PromptForClassification:
    return get_content_grading_model()


@pytest.fixture(scope="session")
def user_answer_df(
    path: str = "app/static/changed_user_answer.csv",
) -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def keyword_runnable() -> KeywordPredictRunnable:
    return KeywordPredictRunnable()


@pytest.fixture(scope="session")
def keyword_data_factory(keyword_model: SentenceTransformer) -> KeywordDataFactory:
    keyword_data_factory = KeywordDataFactory()
    keyword_data_factory.set_problem_dict(keyword_model)
    return keyword_data_factory


@pytest.fixture(scope="session")
def content_data_factory() -> ContentDataFactory:
    return ContentDataFactory()


@pytest.fixture
def random_multi_candidate_keyword_data(keyword_data_factory: KeywordDataFactory) -> KeywordGradingRequest:
    return keyword_data_factory.get_multi_candidate_keyword_request_data()


@pytest.fixture
def random_content_data(content_data_factory: ContentDataFactory) -> ContentGradingRequest:
    return content_data_factory.get_request_data()


@pytest.fixture
def random_keyword_data(keyword_data_factory: KeywordDataFactory) -> KeywordGradingRequest:
    return keyword_data_factory.get_request_data()


@pytest.fixture
def problem_dict(keyword_data_factory: KeywordDataFactory) -> dict:
    return keyword_data_factory.get_problem_dict()
