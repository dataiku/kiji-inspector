from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.dependencies import get_engine


@asynccontextmanager
async def lifespan(app):
    # Finish the checkpoint download and load before accepting requests/probes.
    get_engine()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="SAE Feature Description Server", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
