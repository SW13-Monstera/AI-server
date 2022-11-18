from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from app import config as settings
from app.api.dependency import init_model
from app.api.v1.endpoint import router
from app.exceptions import APIException, APIExceptionErrorCodes, APIExceptionTypes

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    init_model()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=APIExceptionErrorCodes.SCHEMA_ERROR[1],
        content={
            "error": {
                "message": "schema error. please refer to data for details",
                "type": APIExceptionTypes.DATA_VALIDATION,
                "code": APIExceptionErrorCodes.SCHEMA_ERROR[0],
                "data": exc.errors(),
            }
        },
    )


@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.get_exception_content().dict())
