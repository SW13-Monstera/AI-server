from openprompt import PromptForClassification


def test_get_content_model(content_model: PromptForClassification) -> None:
    assert isinstance(content_model, PromptForClassification)
