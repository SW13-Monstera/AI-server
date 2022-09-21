from enum import Enum


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return str.__str__(self)


class GradingStandardEnum(StringEnum):
    CONTENT = "content"
    KEYWORD = "keyword"
