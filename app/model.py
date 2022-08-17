import bentoml
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Huffon/sentence-klue-roberta-base")
bentoml.pytorch.save_model("sentence-ko-roberta", model)
