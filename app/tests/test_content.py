# from openprompt import PromptForClassification
#
# from app.runnable import ContentPredictRunnable
# from app.schemas import ContentGradingRequest, ContentGradingResponse, ContentResponse
#
#
# def test_get_content_model(content_model: PromptForClassification) -> None:
#     assert isinstance(content_model, PromptForClassification)
#
#
# def test_content_predict_runnable(random_content_data: ContentGradingRequest) -> None:
#     runnable = ContentPredictRunnable()
#     assert runnable.model is not None, "runnable의 model이 None입니다."
#
#     result = runnable.is_correct_content(input_data=random_content_data)
#     assert isinstance(result, ContentGradingResponse), "result는 ContentGradingResponse 타입이어야 합니다."
#     assert result.problem_id == random_content_data.problem_id, "problem id 가 일치하지 않습니다."
#     assert isinstance(result.correct_contents, list), "correct_contents 는 list 타입이어야 합니다."
#     for correct_content in result.correct_contents:
#         assert isinstance(correct_content, ContentResponse), "correct_contents 의 원소는 ContentResponse 타입이어야 합니다."
