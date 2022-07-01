
from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact
from random import randrange
from torch import nn

@env(infer_pip_packages=True)  # 예측 서비스 코드에서 사용하는 모든 Pypi 패키지를 자동으로 찾아서 해당 버전 고정
@artifacts([PytorchModelArtifact('model')])  # 예측 서비스와 함께 패킹되는데 필요한 훈련된 모델을 정의
class ProblemSimilarity(BentoService):
    """
    BentoService를 상속받으면 해당 서비스를 Yatai(모델 이미지 레지스트리)에 저장한다.
    """
    @api()  # Todo: add input validator
    def predict(self, user_answer: str, real_answer):
        return randrange(0, 10)


class FakeModel:
    ...


if __name__ == '__main__':
    import torch

    bento_service = ProblemSimilarity()
    device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
    model = FakeModel()
    bento_service.pack("model", model)
    saved_path = bento_service.save()
    print(saved_path)

