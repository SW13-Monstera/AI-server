import bentoml
from bentoml._internal.server.service_app import ServiceAppFactory
from bentoml.io import JSON

from app.runnable import ContentPredictRunnable, KeywordPredictRunnable
from app.schemas import (
    ContentGradingRequest,
    IntegratedGradingRequest,
    IntegratedGradingResponse,
    KeywordGradingRequest,
    KeywordGradingResponse,
)
from app.utils.monkey_patch import _create_api_endpoint

ServiceAppFactory._create_api_endpoint = _create_api_endpoint


keyword_runner = bentoml.Runner(KeywordPredictRunnable)
content_runner = bentoml.Runner(ContentPredictRunnable)

grading_service = bentoml.Service(name="grading_service", runners=[keyword_runner, content_runner])


@grading_service.api(
    input=JSON(pydantic_model=KeywordGradingRequest),
    output=JSON(pydantic_model=KeywordGradingResponse),
)
async def keyword_predict(input_data: KeywordGradingRequest) -> KeywordGradingResponse:
    return await keyword_runner.is_correct_keyword.async_run(input_data)


@grading_service.api(
    input=JSON(pydantic_model=IntegratedGradingRequest),
    output=JSON(pydantic_model=IntegratedGradingResponse),
)
async def integrate_predict(input_data: IntegratedGradingRequest) -> IntegratedGradingResponse:

    keyword_predict_input = KeywordGradingRequest(
        problem_id=input_data.problem_id,
        user_answer=input_data.user_answer,
        keyword_standards=input_data.keyword_standards,
    )
    content_predict_input = ContentGradingRequest(
        problem_id=input_data.problem_id,
        user_answer=input_data.user_answer,
        content_standards=input_data.content_standards,
    )

    content_grading_result = await content_runner.is_correct_content.async_run(content_predict_input)
    keyword_grading_result = await keyword_runner.is_correct_keyword.async_run(keyword_predict_input)

    return IntegratedGradingResponse(
        problem_id=keyword_grading_result.problem_id,
        correct_keywords=keyword_grading_result.correct_keywords,
        correct_contents=content_grading_result.correct_contents,
    )
