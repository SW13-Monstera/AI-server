from typing import List

from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

from app.enums import GradingStandardEnum
from app.exceptions import APIException, APIExceptionErrorCodes, APIExceptionTypes


class GradingStandard(BaseModel):
    id: int = Field(title="아이디")
    content: str = Field(title="채점 기준 내용")

    @validator("content")
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if value == "":
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Grading standard content cannot be empty",
                data=value,
            )
        return value


class KeywordStandard(GradingStandard):
    type = GradingStandardEnum.KEYWORD


class ContentStandard(GradingStandard):
    type = GradingStandardEnum.CONTENT


class Problem(BaseModel):
    keyword_standards: List[KeywordStandard]
    embedded_keywords: NDArray

    @validator("keyword_standards")
    def validate_keywords(cls, value: List[KeywordStandard]) -> List[KeywordStandard]:
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
        if len(values["keyword_standards"]) != value.shape[0]:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Keywords and embedded_keywords must have the same length",
                data=value,
            )
        return value

    class Config:
        arbitrary_types_allowed = True


class UserAnswer(BaseModel):
    problem_id: int = Field(title="문제 아이디")
    user_answer: str = Field(title="유저 답변")


class KeywordGradingRequest(UserAnswer):
    keyword_standards: List[KeywordStandard] = Field(title="키워드 채점 기준 리스트")

    @validator("keyword_standards")
    def validate_grading_standards(cls, value: List[KeywordStandard]) -> List[KeywordStandard]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="keyword standards cannot be empty",
                data=value,
            )
        return value


class KeywordResponse(BaseModel):
    id: int
    keyword: str
    predict_keyword_position: List[int]
    predict_keyword: str


class KeywordGradingResponse(BaseModel):
    problem_id: int
    correct_keywords: List[KeywordResponse]


class IntegratedGradingRequest(KeywordGradingRequest):
    content_standards: List[ContentStandard]

    @validator("content_standards")
    def validate_grading_standards(cls, value: List[ContentStandard]) -> List[ContentStandard]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="content standards cannot be empty",
                data=value,
            )
        return value


class IntegratedGradingResponse(KeywordGradingResponse):
    correct_content_ids: List[int]
