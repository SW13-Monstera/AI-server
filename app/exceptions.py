from typing import Any

from pydantic import BaseModel


class ExceptionDetail(BaseModel):
    message: str
    type: str
    code: str
    data: Any


class APIExceptionSchema(BaseModel):
    error: ExceptionDetail


class APIExceptionErrorCodes:
    BAD_REQUEST = ("bad_request", 400)
    OBJECT_NOT_FOUND = ("not_found", 404)
    SCHEMA_ERROR = ("schema_error", 422)
    INTERNAL_ERROR = ("internal_error", 500)
    NOT_IMPLEMENTED = ("not_implemented", 501)
    BAD_GATEWAY = ("bad_gateway", 502)
    UNAVAILABLE = ("unavailable", 503)
    GATEWAY_TIMEOUT = ("gateway_timeout", 504)


class APIExceptionTypes:
    DATA_VALIDATION = "data_validation"
    INVALID_REQUEST = "invalid_request"


class APIException(Exception):
    def __init__(
        self,
        exception_code: tuple,
        error_type: str = APIExceptionTypes.INVALID_REQUEST,
        message: str = "",
        data: Any = None,
    ):
        self.error_code, self.status_code = exception_code
        self.type = error_type
        self.message = message
        self.data = data

    def get_exception_content(self) -> APIExceptionSchema:
        content = {
            "error": {
                "message": self.message,
                "type": self.type,
                "code": self.error_code,
                "data": [] if self.data is None else self.data,
            }
        }

        return APIExceptionSchema(**content)
