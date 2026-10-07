from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.demo import router as demo_router
from app.dependencies import get_engine


@asynccontextmanager
async def lifespan(app):
    # Finish the checkpoint download and load before accepting requests/probes.
    get_engine()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="SAE Feature Description Server", lifespan=lifespan)
    app.include_router(router)
    app.include_router(demo_router)
    app.mount(
        "/demo/assets", StaticFiles(directory=Path(__file__).parent / "static"), name="demo-assets"
    )
    return app


app = create_app()
