from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="SAE Feature Description Server")
    app.include_router(router)
    return app


app = create_app()
