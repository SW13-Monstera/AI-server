import pandas as pd
import pytest
from openprompt import PromptForClassification
from sentence_transformers import SentenceTransformer

from app.api.dependency import ApplicationContext
from app.controller.content import ContentController
from app.controller.keyeword import KeywordController
from app.schemas import ContentGradingRequest, KeywordGradingRequest
from app.tests.factory import ContentDataFactory, KeywordDataFactory


@pytest.fixture(scope="session")
def user_answer_df(path: str = "app/static/changed_user_answer.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def context() -> ApplicationContext:
    return ApplicationContext()

@pytest.fixture(scope="session")
def keyword_model(context: ApplicationContext) -> SentenceTransformer:
    return context.get_keyword_model()


@pytest.fixture(scope="session")
def content_model(context: ApplicationContext) -> PromptForClassification:
    return context.get_content_model()


@pytest.fixture(scope="session")
def keyword_controller(context: ApplicationContext) -> KeywordController:
    return context.get_keyword_controller()


@pytest.fixture(scope="session")
def content_controller(context: ApplicationContext) -> ContentController:
    return context.get_content_controller()


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
