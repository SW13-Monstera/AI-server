from typing import List

from bentoml.exceptions import BadInput
from numpy.typing import NDArray
from pydantic import BaseModel, validator


class Keyword(BaseModel):
    id: int
    content: str

    @validator("content")
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if value == "":
            raise BadInput("Keyword content cannot be empty")
        return value


class KeywordInferenceRequest(BaseModel):
    problem_id: int
    user_answer: str
    keywords: List[Keyword]

    @validator("user_answer")
    def validate_user_answer(cls, value: str) -> str:
        value = value.strip()
        if value == "":
            raise BadInput("User answer cannot be empty")
        return value

    @validator("keywords")
    def validate_keywords(cls, value: List[Keyword]) -> List[Keyword]:
        if not value:
            raise BadInput("Keywords cannot be empty")
        return value


class Problem(BaseModel):
    keywords: List[Keyword]
    embedded_keywords: NDArray

    @validator("keywords")
    def validate_keywords(cls, value: List[Keyword]) -> List[Keyword]:
        if not value:
            raise BadInput("Keywords cannot be empty")
        return value

    @validator("embedded_keywords")
    def validate_embedded_keywords(cls, value: NDArray, values: dict) -> NDArray:
        if len(values["keywords"]) != value.shape[0]:
            raise BadInput("Keywords and embedded_keywords must have the same length")
        return value

    class Config:
        arbitrary_types_allowed = True


class KeywordResponse(BaseModel):
    id: int
    keyword: str
    predict_keyword_position: List[int]
    predict_keyword: str


class KeywordInferenceResponse(BaseModel):
    problem_id: int
    correct_keywords: List[KeywordResponse]
