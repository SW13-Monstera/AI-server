from typing import List, Optional

from numpy.typing import NDArray
from pydantic import BaseModel


class Keyword(BaseModel):
    id: int
    content: str


class KeywordInferenceData(BaseModel):
    problem_id: int
    user_answer: str
    keywords: List[Keyword]


class Problem(BaseModel):
    subject: Optional[str]
    keywords: List[Keyword]
    embedded_keywords: NDArray

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
