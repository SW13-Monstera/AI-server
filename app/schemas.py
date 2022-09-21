from typing import List

from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

from app.enums import GradingStandardEnum
from app.exceptions import APIException, APIExceptionErrorCodes, APIExceptionTypes


class GradingStandard(BaseModel):
    id: int = Field(title="아이디")
    content: str = Field(title="채점 기준 내용")
    type: GradingStandardEnum = Field(title="채점 기준 타입")

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


class KeywordGradingRequest(BaseModel):
    problem_id: int = Field(title="문제 아이디")
    user_answer: str = Field(title="유저 답변")
    grading_standards: List[GradingStandard] = Field(title="키워드 채점 기준 리스트")

    @validator("grading_standards")
    def validate_grading_standards(cls, value: List[GradingStandard]) -> List[GradingStandard]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="grading standards cannot be empty",
                data=value,
            )
        return value


# class IntegratedGradingRequest(BaseModel):
#     problem_id: int
#     user_answer: str
#     keywords: List[GradingStandard]
#     key_contents:


class Problem(BaseModel):
    keywords: List[GradingStandard]
    embedded_keywords: NDArray

    @validator("keywords")
    def validate_keywords(cls, value: List[GradingStandard]) -> List[GradingStandard]:
        if not value:
            raise APIException(
                exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                error_type=APIExceptionTypes.DATA_VALIDATION,
                message="Keywords cannot be empty",
                data=value,
            )
        for standard in value:
            if standard.type != GradingStandardEnum.KEYWORD:
                raise APIException(
                    exception_code=APIExceptionErrorCodes.SCHEMA_ERROR,
                    error_type=APIExceptionTypes.DATA_VALIDATION,
                    message="Embedded keywords schema has only keyword standard",
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
