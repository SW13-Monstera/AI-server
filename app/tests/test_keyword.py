import random

import pytest

from app.controller.keyeword import KeywordController
from app.schemas import KeywordGradingRequest, KeywordStandard
from app.tests.factory import KeywordDataFactory


@pytest.mark.asyncio
async def test_keyword_predict_runnable(
    random_keyword_data: KeywordGradingRequest, keyword_controller: KeywordController
) -> None:
    """
    로컬 메모리에 problem 정보가 없을 때 새롭게 problem에 대한 키워드 정보를 생성하고 예측한다.
    """
    test_problem_id = random_keyword_data.problem_id
    assert keyword_controller.problem_dict == {}, "init problem dict는 빈 dict 상태입니다."
    result = await keyword_controller.grading(random_keyword_data)

    runnable_problem = keyword_controller.problem_dict[test_problem_id]
    assert test_problem_id in keyword_controller.problem_dict, "new problem 정보가 업데이트 되지 않았습니다."

    for runnable_keyword in runnable_problem.keyword_standards:
        success = False
        for test_keyword in random_keyword_data.keyword_standards:
            if runnable_keyword.id == test_keyword.id and runnable_keyword.content in test_keyword.content:
                success = True
                break

        assert success, "runnable 내부의 problem_dict에 동기화가 되지 않았습니다."

    assert test_problem_id == result.problem_id, "request와 response의 문제 ID가 같지 않습니다."


@pytest.mark.asyncio
async def test_keyword_predict_runnable_2(
    problem_dict: dict, random_keyword_data: KeywordGradingRequest, keyword_controller: KeywordController
) -> None:
    """
    기존에 있던 키워드가 하나 지워지고 새로 생겼을 때 메모리에서도 지우고 새로 생긴 키워드를 임베딩해서 저장한다.
    """
    every_keyword_ids = []
    for problem_id in problem_dict:
        for keyword in problem_dict[problem_id].keyword_standards:
            every_keyword_ids.append(keyword.id)
    problem_id = random_keyword_data.problem_id
    new_keyword = KeywordStandard(id=max(every_keyword_ids) + 1, content="new keyword")
    test_problem_id = random_keyword_data.problem_id
    delete_keyword = random_keyword_data.keyword_standards.pop()  # 키워드 하나 삭제
    random_keyword_data.keyword_standards.append(new_keyword)  # 키워드 하나 추가
    result = await keyword_controller.grading(random_keyword_data)

    problem_keyword_id_set = set(
        keyword.id for keyword in keyword_controller.problem_dict[test_problem_id].keyword_standards
    )
    assert delete_keyword.id not in problem_keyword_id_set, "삭제되어야 할 키워드가 메모리에서 지워지지 않았습니다."
    assert new_keyword.id in problem_keyword_id_set, "새로 생긴 키워드가 메모리에 저장되지 않았습니다."
    assert keyword_controller.problem_dict[problem_id].embedded_keywords.shape[0] == len(
        keyword_controller.problem_dict[problem_id].keyword_standards
    ), "키워드 임베딩이 제대로 저장되지 않았습니다."

    runnable_problem = keyword_controller.problem_dict[problem_id]
    for runnable_keyword in runnable_problem.keyword_standards:
        success = False
        for test_keyword in random_keyword_data.keyword_standards:
            if runnable_keyword.id == test_keyword.id and runnable_keyword.content in test_keyword.content:
                success = True

        assert success, "runnable 내부의 problem_dict에 동기화가 되지 않았습니다."

    assert test_problem_id == result.problem_id, "request와 response의 문제 ID가 같지 않습니다."

    problem_keyword_set = set(
        keyword.id for keyword in keyword_controller.problem_dict[test_problem_id].keyword_standards
    )
    for correct_keyword in result.correct_keywords:
        assert correct_keyword.id in problem_keyword_set, "problem_dict에 맞지 않는 키워드를 이용해 예측하였습니다."


@pytest.mark.asyncio
async def test_keyword_predict_runnable_3(
    random_multi_candidate_keyword_data: KeywordGradingRequest, keyword_controller: KeywordController
) -> None:
    """
    ','로 구분되어 있는 키워드 기준들을 함께 보면서 유사도 측정
    """
    standard_with_comma = ""
    for standard in random_multi_candidate_keyword_data.keyword_standards:
        if ", " in standard.content:
            standard_with_comma = standard.content
    random_multi_candidate_keyword_data.user_answer += standard_with_comma
    response = await keyword_controller.grading(random_multi_candidate_keyword_data)
    assert standard_with_comma in (keyword.keyword for keyword in response.correct_keywords)


@pytest.mark.asyncio
async def test_keyword_predict_runnable_4(
    problem_dict: dict,
    random_multi_candidate_keyword_data: KeywordGradingRequest,
    keyword_controller: KeywordController,
) -> None:
    """
    새롭게 ','가 포함된 키워드가 추가되었을 때 update 되는지 테스트
    """
    id_set_1 = set(map(lambda x: x.id, problem_dict[random_multi_candidate_keyword_data.problem_id].keyword_standards))
    id_set_2 = set(map(lambda x: x.id, random_multi_candidate_keyword_data.keyword_standards))
    assert id_set_1 == id_set_2, "테스트 데이터의 설계가 잘못 되었습니다. keyword의 id들을 확인해주세요"

    test_candidate_keywords = []
    for keyword_standard in random_multi_candidate_keyword_data.keyword_standards:
        if ", " in keyword_standard.content:
            test_candidate_keywords.append(keyword_standard)
    test_keywords = random.choice(test_candidate_keywords)
    test_keywords.id = max(id_set_1) + 1
    test_keywords.content = "test, data, 입니다"

    await keyword_controller.grading(random_multi_candidate_keyword_data)
    runnable_problem = keyword_controller.problem_dict[random_multi_candidate_keyword_data.problem_id]
    assert (
        len(runnable_problem.keyword_standards) == runnable_problem.embedded_keywords.shape[0]
    ), "업데이트된 Problem의 shape 정보가 맞지 않습니다."
    for runnable_keyword in runnable_problem.keyword_standards:
        success = False
        for test_keyword in random_multi_candidate_keyword_data.keyword_standards:
            if runnable_keyword.id == test_keyword.id and runnable_keyword.content in test_keyword.content:
                success = True

        assert success, "runnable 내부의 problem_dict에 동기화가 되지 않았습니다."


@pytest.mark.asyncio
async def test_empty_tokenized_answer_test(
    keyword_data_factory: KeywordDataFactory, keyword_controller: KeywordController
) -> None:
    """
    tokenizing 했을 때 비어 있는 list 라면 predict 없이 바로 return
    """
    bad_input = "ㅁㄴㅇ레ㅐㅓㅁㄴㅇ레"
    tokenized_answer = keyword_controller.get_tokenized_answer(bad_input)
    assert tokenized_answer == []
    request_data = keyword_data_factory.get_request_data()
    request_data.user_answer = bad_input
    response_data = await keyword_controller.grading(request_data)
    assert response_data.correct_keywords == []
