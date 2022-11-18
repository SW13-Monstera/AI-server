from typing import List

from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

from app.enums import GradingStandardEnum
from app.exceptions import APIException, APIExceptionErrorCodes, APIExceptionTypes
from app.utils.utils import (
    get_integrated_grading_request_example,
    get_integrated_grading_response_example,
    get_keyword_grading_request_example,
    get_keyword_grading_response_example,
)


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
    keyword_standards: List[KeywordStandard] = Field("키워드 채점 기준들")
    embedded_keywords: NDArray = Field("키워드 채점 기준 임베딩")

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

    class Config:
        schema_extra = {"example": get_keyword_grading_request_example()}


class KeywordSimilarityInfo(KeywordStandard):
    score: float = Field(title="유사도 점수", description="점수는 0에서 1사이의 값")
    tokenized_answer_idx: int = Field(title="토크나이징된 유저 답변의 인덱스")


class ContentGradingRequest(UserAnswer):
    content_standards: List[ContentStandard] = Field(title="내용 채점 기준 리스트")

    @validator("keyword_standards", check_fields=False)
    def validate_grading_standards(cls, value: List[ContentStandard]) -> List[ContentStandard]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="content standards cannot be empty",
                data=value,
            )
        return value


class KeywordResponse(BaseModel):
    id: int = Field(title="키워드 채점 기준 아이디")
    keyword: str = Field(title="키워드")
    predict_keyword_position: List[int] = Field(title="모델이 예측한 키워드 인덱스")
    predict_keyword: str = Field(title="모델이 예측한 키워드")


class KeywordGradingResponse(BaseModel):
    problem_id: int = Field(title="문제 아이디")
    correct_keywords: List[KeywordResponse] = Field(title="예상 키워드 리스트")

    class Config:
        schema_extra = {"example": get_keyword_grading_response_example()}


class ContentResponse(BaseModel):
    id: int = Field(title="내용 채점 기준 아이디")
    content: str = Field(title="내용")


class ContentGradingResponse(BaseModel):
    problem_id: int = Field(title="문제 아이디")
    correct_contents: List[ContentResponse] = Field(title="예상 핵심 내용 아이디 리스트")


class IntegratedGradingRequest(KeywordGradingRequest, ContentGradingRequest):
    class Config:
        schema_extra = {"example": get_integrated_grading_request_example()}


class IntegratedGradingResponse(KeywordGradingResponse, ContentGradingResponse):
    class Config:
        schema_extra = {"example": get_integrated_grading_response_example()}
