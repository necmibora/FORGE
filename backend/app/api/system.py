from fastapi import APIRouter

from app.schemas import SystemInfo
from app.services.gpu_info import collect_system_info

router = APIRouter(tags=["system"])


@router.get("/system", response_model=SystemInfo)
def get_system() -> SystemInfo:
    return collect_system_info()
