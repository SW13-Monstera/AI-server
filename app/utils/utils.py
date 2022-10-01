import os

import numpy as np
import pandas as pd


def update_problem_info(problem_info_path: str, user_answer_path: str):

    problem_info_df = pd.read_csv(problem_info_path)

    problem_info = {}
    for i in range(len(problem_info_df)):
        row = problem_info_df.iloc[i]
        if row.problem_id not in problem_info:
            problem_info[row.problem_id] = {"keyword_standards": [], "content_standards": []}
        if row.content == np.nan:
            row.content = "NULL"
        if row.type == "KEYWORD":
            problem_info[row.problem_id]["keyword_standards"].append(row.content)
        elif row.type == "PROMPT":
            problem_info[row.problem_id]["content_standards"].append(row.content)

    user_answer_df = pd.read_csv(user_answer_path)

    change_dict = {4: 449, 6: 453, 1: 454, 5: 450, 9: 447, 3: 451, 8: 448, 2: 446, 0: 452, 7: 443}
    visit = set()
    keyword_remove_dict = {}
    content_remove_dict = {}
    for i in range(len(user_answer_df)):
        problem_id = change_dict[user_answer_df.iloc[i].problem_id]
        converted_problem = problem_info[problem_id]
        if problem_id not in visit:
            print(user_answer_df.keyword_criterion[i])
            print(converted_problem["keyword_standards"])
            order = list(map(int, input().split()))
            if len(eval(user_answer_df.keyword_criterion[i])) != len(converted_problem["keyword_standards"]):
                print("삭제할 인덱스를 적어주세요")
                keyword_remove_dict[user_answer_df.iloc[i].problem_id] = list(map(int, input().split()))

            converted_problem["keyword_standards"] = [converted_problem["keyword_standards"][j - 1] for j in order]
            print(user_answer_df.keyword_criterion[i])
            print(converted_problem["keyword_standards"])
            print("*" * 50)
            print(user_answer_df.scoring_criterion[i])
            print(converted_problem["content_standards"])
            order = list(map(int, input().split()))
            if len(eval(user_answer_df.scoring_criterion[i])) != len(converted_problem["content_standards"]):
                print("삭제할 인덱스를 적어주세요")
                content_remove_dict[user_answer_df.iloc[i].problem_id] = list(map(int, input().split()))

            converted_problem["content_standards"] = [converted_problem["content_standards"][j - 1] for j in order]
            print(user_answer_df.scoring_criterion[i])
            print(converted_problem["content_standards"])
            print("*" * 50)
            visit.add(problem_id)
        labeled_keyword_ids = []
        correct_keyword_criterion = eval(user_answer_df.correct_keyword_criterion[i])
        keyword_criterion = eval(user_answer_df.keyword_criterion[i])
        if user_answer_df.iloc[i].problem_id in keyword_remove_dict:
            for idx in keyword_remove_dict[user_answer_df.iloc[i].problem_id]:
                keyword_criterion[idx - 1] = None
        keyword_criterion = [value for value in keyword_criterion if value is not None]
        keyword_criterion = [value if value != np.nan else "NULL" for value in keyword_criterion]
        print(correct_keyword_criterion)
        print(keyword_criterion)

        for criterion in correct_keyword_criterion:
            if criterion in keyword_criterion:
                idx = keyword_criterion.index(criterion)
                labeled_keyword_ids.append(idx)
        print(labeled_keyword_ids)
        print(converted_problem["keyword_standards"])
        labeled_content_ids = []
        correct_scoring_criterion = eval(user_answer_df.correct_scoring_criterion[i])
        scoring_criterion = eval(user_answer_df.scoring_criterion[i])

        if user_answer_df.iloc[i].problem_id in content_remove_dict:
            for idx in content_remove_dict[user_answer_df.iloc[i].problem_id]:
                scoring_criterion[idx - 1] = None
        scoring_criterion = [value for value in scoring_criterion if value is not None]

        print(scoring_criterion)
        print(correct_scoring_criterion)
        for criterion in correct_scoring_criterion:
            if criterion in scoring_criterion:
                idx = scoring_criterion.index(criterion)
                labeled_content_ids.append(idx)
        print(labeled_content_ids)
        print(converted_problem["content_standards"])
        print("*" * 50)
        user_answer_df.correct_keyword_criterion[i] = labeled_keyword_ids
        user_answer_df.correct_scoring_criterion[i] = labeled_content_ids
        user_answer_df.keyword_criterion[i] = converted_problem["keyword_standards"]
        user_answer_df.scoring_criterion[i] = converted_problem["content_standards"]

    user_answer_df.to_csv(os.path.join(os.path.dirname(user_answer_path), "changed_user_answer.csv"))


if __name__ == "__main__":
    update_problem_info(
        problem_info_path="/Users/minjaewon/workspace/AI-server/app/static/problem_info.csv",
        user_answer_path="/Users/minjaewon/workspace/AI-server/app/static/user_answer.csv",
    )
