import asyncio

from fastapi import APIRouter, Body, Depends

from app import schemas
from app.api.dependency import get_content_controller, get_keyword_controller
from app.controller.content import ContentController
from app.controller.keyeword import KeywordController
from app.schemas import ContentGradingRequest, IntegratedGradingResponse, KeywordGradingRequest, KeywordGradingResponse

router = APIRouter(prefix="/predict", tags=["grading"])


@router.post("/keyword")
async def keyword_predict(
    keyword_grading_req: KeywordGradingRequest = Body(...),
    keyword_controller: KeywordController = Depends(get_keyword_controller),
) -> KeywordGradingResponse:
    return await keyword_controller.grading(keyword_grading_req)


@router.post("/integrate")
async def integrate_predict(
    integrated_grading_req: schemas.IntegratedGradingRequest = Body(...),
    keyword_controller: KeywordController = Depends(get_keyword_controller),
    content_controller: ContentController = Depends(get_content_controller),
) -> schemas.IntegratedGradingResponse:
    keyword_predict_input = KeywordGradingRequest(
        problem_id=integrated_grading_req.problem_id,
        user_answer=integrated_grading_req.user_answer,
        keyword_standards=integrated_grading_req.keyword_standards,
    )
    content_predict_input = ContentGradingRequest(
        problem_id=integrated_grading_req.problem_id,
        user_answer=integrated_grading_req.user_answer,
        content_standards=integrated_grading_req.content_standards,
    )

    keyword_grading_result, content_grading_result = await asyncio.gather(
        keyword_controller.grading(keyword_predict_input),
        content_controller.grading(content_predict_input),
    )
    return IntegratedGradingResponse(
        problem_id=keyword_grading_result.problem_id,
        correct_keywords=keyword_grading_result.correct_keywords,
        correct_contents=content_grading_result.correct_contents,
    )
