# import logging
# import re
# from pprint import pformat
# from typing import List, Optional, Tuple
#
# import bentoml
# import torch.cuda
# from konlpy.tag import Mecab
# from numpy.typing import NDArray
# from openprompt import PromptDataLoader
# from openprompt.data_utils import InputExample
# from openprompt.plms import T5TokenizerWrapper
# from sklearn.metrics.pairwise import cosine_similarity
#
# from app.config import MECAB_DIC_PATH, OS
# from app.model import get_content_grading_model, get_keyword_grading_model
# from app.schemas import (
#     ContentGradingRequest,
#     ContentGradingResponse,
#     ContentResponse,
#     KeywordGradingRequest,
#     KeywordGradingResponse,
#     KeywordResponse,
#     KeywordSimilarityInfo,
#     KeywordStandard,
#     Problem,
# )
# from app.utils.utils import get_stopwords
#
# log = logging.getLogger("__main__")
#
#
# if OS == "Windows":
#     import win32file
#
#     win32file._setmaxstdio(2048)
#
#
#
#
#
#
