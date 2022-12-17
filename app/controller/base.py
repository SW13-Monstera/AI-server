from abc import ABC, abstractmethod

from app.schemas import UserAnswer


class BaseController(ABC):
    @abstractmethod
    async def grading(self, input_data: UserAnswer):
        pass
