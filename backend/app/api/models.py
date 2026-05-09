from fastapi import APIRouter, HTTPException

from app.schemas import LoadModelRequest, LoadedModelStatus, ModelList
from app.services.model_registry import list_models
from app.services.vllm_runner import runner

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelList)
def get_models() -> ModelList:
    return ModelList(models=list_models())


@router.get("/models/loaded", response_model=LoadedModelStatus)
def get_loaded() -> LoadedModelStatus:
    return LoadedModelStatus(
        loaded=runner.loaded,
        path=runner.loaded_path,
        quantization=runner.loaded_quant,
        inference_available=runner.inference_available,
    )


@router.post("/models/load", response_model=LoadedModelStatus)
async def load_model(req: LoadModelRequest) -> LoadedModelStatus:
    try:
        await runner.load(req)
    except ImportError as e:
        return LoadedModelStatus(
            loaded=False,
            inference_available=False,
            inference_message=f"vLLM not available: {e}",
        )
    except Exception as e:
        if not runner.inference_available:
            return LoadedModelStatus(
                loaded=False,
                inference_available=False,
                inference_message=f"Inference unavailable: {e}",
            )
        raise HTTPException(500, f"Failed to load model: {e}")
    return LoadedModelStatus(
        loaded=True,
        path=runner.loaded_path,
        quantization=runner.loaded_quant,
        inference_available=True,
    )


@router.post("/models/unload", response_model=LoadedModelStatus)
async def unload_model() -> LoadedModelStatus:
    await runner.unload()
    return LoadedModelStatus(
        loaded=False,
        inference_available=runner.inference_available,
    )
