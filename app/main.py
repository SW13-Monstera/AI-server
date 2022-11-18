from fastapi import FastAPI

from app import config as settings
from app.api.v1.endpoint import router

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(router)
print("ready")
