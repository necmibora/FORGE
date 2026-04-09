from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import system, models, inference, benchmarks

app = FastAPI(title="FORGE Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(benchmarks.router)


@app.get("/")
def root():
    return {
        "name": "FORGE",
        "version": "0.1.0",
        "models_dir": str(settings.models_dir),
        "quant_dir": str(settings.quant_dir),
    }
