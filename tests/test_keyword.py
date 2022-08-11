import pytest
from bentoml import Runner
from bentoml.testing.utils import async_request

from schemas import Keyword, KeywordInferenceData
from service import KeywordPredictRunnable


def test_debug(
    keyword_runner: Runner,
    problem_dict: dict,
    random_keyword_data: KeywordInferenceData,
):

    keyword_runner.init_local()

    keyword_runner.is_correct_keyword.run(random_keyword_data)
    print()


@pytest.mark.asyncio
async def test_keyword_inference(
    host: str, random_keyword_data: KeywordInferenceData
) -> None:
    """
    api response 200 test
    """
    await async_request(
        method="POST",
        url=f"http://{host}/keyword_predict",
        headers={"Content-Type": "application/json"},
        data=random_keyword_data.json(),
        assert_status=200,
    )


def test_keyword_predict_runnable(
    problem_dict: dict, random_keyword_data: KeywordInferenceData
) -> None:
    """
    로컬 메모리에 problem 정보가 없을 때 새롭게 problem에 대한 키워드 정보를 생성하고 예측한다.
    """
    test_problem_id = random_keyword_data.problem_id
    runnable = KeywordPredictRunnable(problem_dict)
    if test_problem_id in runnable.problem_dict:
        runnable.problem_dict.pop(random_keyword_data.problem_id)

    assert test_problem_id not in runnable.problem_dict
    result = runnable.is_correct_keyword(random_keyword_data)
    assert test_problem_id in runnable.problem_dict
    assert (
        random_keyword_data.keywords == runnable.problem_dict[test_problem_id].keywords
    )
    assert test_problem_id == result.problem_id

    problem_keyword_set = set(
        keyword.id for keyword in problem_dict[random_keyword_data.problem_id].keywords
    )
    for correct_keyword in result.correct_keywords:
        assert correct_keyword.id in problem_keyword_set


def test_keyword_predict_runnable_2(
    problem_dict: dict, random_keyword_data: KeywordInferenceData
) -> None:
    """
    기존에 있던 키워드가 하나 지워지고 새로 생겼을 때 메모리에서도 지우고 새로 생긴 키워드를 임베딩해서 저장한다.
    """
    every_keyword_ids = []
    for problem_id in problem_dict:
        for keyword in problem_dict[problem_id].keywords:
            every_keyword_ids.append(keyword.id)
    new_keyword = Keyword(id=max(every_keyword_ids) + 1, content="new keyword")
    test_problem_id = random_keyword_data.problem_id
    runnable = KeywordPredictRunnable(problem_dict)
    delete_keyword = random_keyword_data.keywords.pop()  # 키워드 하나 삭제
    random_keyword_data.keywords.append(new_keyword)  # 키워드 하나 추가
    result = runnable.is_correct_keyword(random_keyword_data)

    problem_keyword_id_set = set(
        keyword.id for keyword in runnable.problem_dict[test_problem_id].keywords
    )
    assert (
        delete_keyword.id not in problem_keyword_id_set
    ), "삭제되어야 할 키워드가 메모리에서 지워지지 않았습니다."
    assert new_keyword.id in problem_keyword_id_set, "새로 생긴 키워드가 메모리에 저장되지 않았습니다."

    assert (
        random_keyword_data.keywords == runnable.problem_dict[test_problem_id].keywords
    ), "키워드 동기화가 되지 않았습니다."
    assert test_problem_id == result.problem_id, "request와 response의 문제 ID가 같지 않습니다."

    problem_keyword_set = set(
        keyword.id for keyword in runnable.problem_dict[test_problem_id].keywords
    )
    for correct_keyword in result.correct_keywords:
        assert correct_keyword.id in problem_keyword_set
