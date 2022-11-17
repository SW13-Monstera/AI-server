# import random
# from abc import ABC, abstractmethod
# from copy import deepcopy
# from typing import List
#
# import pandas as pd
# from sentence_transformers import SentenceTransformer
#
# from app.config import root
# from app.schemas import ContentGradingRequest, ContentStandard, KeywordGradingRequest, KeywordStandard, Problem
#
#
# class ProblemFactory(ABC):
#     def __init__(self, csv_path: str = f"{root}/app/static/changed_user_answer.csv"):
#         self.df = pd.read_csv(csv_path)
#         self.df.keyword_criterion = self.df.keyword_criterion.str.replace("nan", "'NULL'").map(eval)
#         self.df.scoring_criterion = self.df.scoring_criterion.map(eval)
#         self.problem_dict = {}
#
#     @abstractmethod
#     def get_request_data(self):
#         pass
#
#     @abstractmethod
#     def get_many_request_data(self, k: int):
#         pass
#
#     def get_random_series(self) -> pd.Series:
#         random_idx = random.randint(0, len(self.df) - 1)
#         random_series = self.df.iloc[random_idx]
#         return random_series
#
#
# class KeywordDataFactory(ProblemFactory):
#     def get_request_data(self) -> KeywordGradingRequest:
#         random_series = self.get_random_series()
#         keyword_standards = []
#         offset = random.randint(10001, 20000)
#         for i, keyword_content in enumerate(random_series.keyword_criterion):
#             keyword_standards.append(KeywordStandard(id=offset + i, content=keyword_content))
#         return KeywordGradingRequest(
#             problem_id=random_series.problem_id,
#             user_answer=random_series.user_answer,
#             keyword_standards=keyword_standards,
#         )
#
#     def get_many_request_data(self, k: int) -> List[KeywordGradingRequest]:
#         return [self.get_request_data() for _ in range(k)]
#
#     def get_multi_candidate_keyword_request_data(self) -> KeywordGradingRequest:
#         search = True
#         while search:
#             random_idx = random.randint(0, len(self.df) - 1)
#             random_data = self.df.iloc[random_idx]
#             problem_id = random_data.problem_id
#             err_message = f"static data file에 무결성이 깨졌습니다. problem : {problem_id}"
#             keyword_contents = random_data.keyword_criterion
#             assert self.keyword_valid_check(problem_id, keyword_contents), err_message
#             search = not any(filter(lambda x: ", " in x, keyword_contents))
#
#         keyword_id = problem_id * 10 + 1
#         keyword_standards = [
#             KeywordStandard(id=keyword_id + i, content=content) for i, content in enumerate(keyword_contents)
#         ]
#         return KeywordGradingRequest(
#             problem_id=problem_id, user_answer=random_data.user_answer, keyword_standards=keyword_standards
#         )
#
#     def set_problem_dict(self, keyword_model: SentenceTransformer):
#         problem_dict = {}
#
#         for _, row in self.df.iterrows():
#             problem_id = row.problem_id
#             keyword_id = problem_id * 10
#             if problem_id not in problem_dict:
#                 keyword_standards = []
#                 for content in row.keyword_criterion:
#                     keyword_id += 1
#
#                     for split_content in content.split(","):
#                         keyword_standards.append(KeywordStandard(id=keyword_id, content=split_content))
#
#                 embedded_keywords = keyword_model.encode([keyword.content for keyword in keyword_standards])
#                 problem_dict[problem_id] = Problem(
#                     keyword_standards=keyword_standards,
#                     embedded_keywords=embedded_keywords,
#                 )
#         self.problem_dict = problem_dict
#
#     def get_problem_dict(self) -> dict:
#         return deepcopy(self.problem_dict)
#
#     def keyword_valid_check(self, problem_id: int, keywords: List[str]) -> bool:
#         standard_contents = set(standard.content for standard in self.problem_dict[problem_id].keyword_standards)
#         for keyword in keywords:
#             for content in keyword.split(", "):
#                 if content not in standard_contents:
#                     return False
#         return True
#
#
# class ContentDataFactory(ProblemFactory):
#     def get_request_data(self) -> ContentGradingRequest:
#         random_series = self.get_random_series()
#         content_standards = []
#         offset = random.randint(0, 10000)
#         for i, content in enumerate(random_series["scoring_criterion"]):
#             content_standards.append(ContentStandard(id=offset + i, content=content))
#
#         return ContentGradingRequest(
#             problem_id=random_series.problem_id,
#             user_answer=random_series.user_answer,
#             content_standards=content_standards,
#         )
#
#     def get_many_request_data(self, k: int) -> List[ContentGradingRequest]:
#         return [self.get_request_data() for _ in range(k)]
