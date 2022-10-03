from app.schemas import KeywordGradingRequest, KeywordStandard
from app.service import KeywordPredictRunnable


def test_keyword_predict_runnable(
    keyword_runnable: KeywordPredictRunnable, random_keyword_data: KeywordGradingRequest
) -> None:
    """
    로컬 메모리에 problem 정보가 없을 때 새롭게 problem에 대한 키워드 정보를 생성하고 예측한다.
    """
    test_problem_id = random_keyword_data.problem_id
    assert keyword_runnable.problem_dict == {}, "init problem dict는 빈 dict 상태입니다."
    result = keyword_runnable.is_correct_keyword(random_keyword_data)
    assert test_problem_id in keyword_runnable.problem_dict, "new problem 정보가 업데이트 되지 않았습니다."
    assert (
        random_keyword_data.keyword_standards == keyword_runnable.problem_dict[test_problem_id].keyword_standards
    ), "키워드 동기화가 되지 않았습니다."
    assert test_problem_id == result.problem_id, "request와 response의 문제 ID가 같지 않습니다."


def test_keyword_predict_runnable_2(problem_dict: dict, random_keyword_data: KeywordGradingRequest) -> None:
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
    runnable = KeywordPredictRunnable(problem_dict)
    delete_keyword = random_keyword_data.keyword_standards.pop()  # 키워드 하나 삭제
    random_keyword_data.keyword_standards.append(new_keyword)  # 키워드 하나 추가
    result = runnable.is_correct_keyword(random_keyword_data)

    problem_keyword_id_set = set(keyword.id for keyword in runnable.problem_dict[test_problem_id].keyword_standards)
    assert delete_keyword.id not in problem_keyword_id_set, "삭제되어야 할 키워드가 메모리에서 지워지지 않았습니다."
    assert new_keyword.id in problem_keyword_id_set, "새로 생긴 키워드가 메모리에 저장되지 않았습니다."
    assert runnable.problem_dict[problem_id].embedded_keywords.shape[0] == len(
        runnable.problem_dict[problem_id].keyword_standards
    ), "키워드 임베딩이 제대로 저장되지 않았습니다."
    assert (
        random_keyword_data.keyword_standards == runnable.problem_dict[test_problem_id].keyword_standards
    ), "키워드 동기화가 되지 않았습니다."
    assert test_problem_id == result.problem_id, "request와 response의 문제 ID가 같지 않습니다."

    problem_keyword_set = set(keyword.id for keyword in runnable.problem_dict[test_problem_id].keyword_standards)
    for correct_keyword in result.correct_keywords:
        assert correct_keyword.id in problem_keyword_set, "problem_dict에 맞지 않는 키워드를 이용해 예측하였습니다."


def test_keyword_predict_runnable_3(
    keyword_runnable: KeywordPredictRunnable, random_multi_candidate_keyword_data: KeywordGradingRequest
) -> None:
    """
    ','로 구분되어 있는 키워드 기준들을 함께 보면서 유사도 측정
    """
    standard_with_comma = ""
    for standard in random_multi_candidate_keyword_data.keyword_standards:
        if "," in standard.content:
            standard_with_comma = standard.content
    random_multi_candidate_keyword_data.user_answer += standard_with_comma
    response = keyword_runnable.is_correct_keyword(random_multi_candidate_keyword_data)
    assert standard_with_comma in (keyword.keyword for keyword in response.correct_keywords)


def test_keyword_predict_runnable_4(
    problem_dict: dict, random_multi_candidate_keyword_data: KeywordGradingRequest
) -> None:
    """
    새롭게 ','가 포함된 키워드가 추가되었을 때 update 되는지 테스트
    """
