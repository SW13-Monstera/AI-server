import asyncio
import sys
import typing as t

from bentoml._internal.context import InferenceApiContext as Context
from bentoml._internal.io_descriptors.multipart import Multipart
from bentoml._internal.server.service_app import ServiceAppFactory, log_exception
from bentoml._internal.service.inference_api import InferenceAPI
from starlette.requests import Request
from starlette.responses import Response

from app.exceptions import APIException, APIExceptionErrorCodes, APIExceptionTypes


@staticmethod
def _create_api_endpoint(
    api: InferenceAPI,
) -> t.Callable[[Request], t.Coroutine[t.Any, t.Any, Response]]:
    """
    Create api function for flask route, it wraps around user defined API
    callback and adapter class, and adds request logging and instrument metrics
    """
    from starlette.concurrency import run_in_threadpool  # type: ignore
    from starlette.responses import JSONResponse

    async def api_func(request: Request) -> Response:
        # handle_request may raise 4xx or 5xx exception.
        try:
            input_data = await api.input.from_http_request(request)
            ctx = None
            if asyncio.iscoroutinefunction(api.func):
                if isinstance(api.input, Multipart):
                    if api.needs_ctx:
                        ctx = Context.from_http(request)
                        input_data[api.ctx_param] = ctx
                    output = await api.func(**input_data)
                else:
                    if api.needs_ctx:
                        ctx = Context.from_http(request)
                        output = await api.func(input_data, ctx)
                    else:
                        output = await api.func(input_data)
            else:
                if isinstance(api.input, Multipart):
                    if api.needs_ctx:
                        ctx = Context.from_http(request)
                        input_data[api.ctx_param] = ctx
                    output: t.Any = await run_in_threadpool(api.func, **input_data)
                else:
                    if api.needs_ctx:
                        ctx = Context.from_http(request)
                        output = await run_in_threadpool(api.func, input_data, ctx)
                    else:
                        output = await run_in_threadpool(api.func, input_data)

            response = await api.output.to_http_response(output, ctx)

        except APIException as e:
            log_exception(request, sys.exc_info())
            response = JSONResponse(
                content=e.get_exception_content().dict(),
                status_code=e.status_code,
            )
        except Exception:  # pylint: disable=broad-except
            log_exception(request, sys.exc_info())

            response = JSONResponse(
                content={
                    "error": {
                        "message": "An error has occurred in BentoML user code when handling this request, find the error details in server logs",  # noqa
                        "type": APIExceptionTypes.INVALID_REQUEST,
                        "code": APIExceptionErrorCodes.INTERNAL_ERROR[1],
                        "data": [],
                    }
                },
                status_code=APIExceptionErrorCodes.INTERNAL_ERROR[1],
            )
        return response

    return api_func


ServiceAppFactory._create_api_endpoint = _create_api_endpoint
