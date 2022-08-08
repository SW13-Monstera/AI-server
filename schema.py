from pydantic import BaseModel
from typing import List
from numpy.typing import NDArray


class UserAnswer(BaseModel):
    problem_id: int
    user_answer: str


class KeywordPredictData(UserAnswer):
    keywords: List[str]
    embedded_keywords: List[NDArray]  # NDArray 하나로 갈건지 아니면 keywords랑 비슷하게 List안에 NDArray로 묶을지 고민


"""
{
    "correct_keyword": [
        {
            "keyword_id": 1,
            "predict_keyword_position": [22, 27]
        },
        {
            "keyword_id": 3,
            "predict_keyword_position": [37, 42]
        }
        {
            "keyword_id": 4,
            "predict_keyword_position": [10, 19]
        }
    ]
}
"""

