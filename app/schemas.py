from typing import List

from numpy.typing import NDArray
from pydantic import BaseModel, validator

from app.exceptions import APIException, APIExceptionErrorCodes, APIExceptionTypes


class Keyword(BaseModel):
    id: int
    content: str

    @validator("content")
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if value == "":
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Keyword content cannot be empty",
                data=value,
            )
        return value


class KeywordGradingRequest(BaseModel):
    problem_id: int
    user_answer: str
    keywords: List[Keyword]

    @validator("keywords")
    def validate_keywords(cls, value: List[Keyword]) -> List[Keyword]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Keywords cannot be empty",
                data=value,
            )
        return value


class EmbeddedKeywords(BaseModel):
    keywords: List[Keyword]
    embedded_keywords: NDArray

    @validator("keywords")
    def validate_keywords(cls, value: List[Keyword]) -> List[Keyword]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Keywords cannot be empty",
                data=value,
            )
        return value

    @validator("embedded_keywords")
    def validate_embedded_keywords(cls, value: NDArray, values: dict) -> NDArray:
        if len(values["keywords"]) != value.shape[0]:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Keywords and embedded_keywords must have the same length",
                data=value,
            )
        return value

    class Config:
        arbitrary_types_allowed = True


class KeywordResponse(BaseModel):
    id: int
    keyword: str
    predict_keyword_position: List[int]
    predict_keyword: str


class KeywordGradingResponse(BaseModel):
    problem_id: int
    correct_keywords: List[KeywordResponse]
