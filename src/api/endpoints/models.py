from fastapi import APIRouter
router = APIRouter(prefix="/api/v1/models", tags=["models"])
@router.get("/active")
def active_model() -> dict:
    from ..main import app_state
    metrics = app_state.service.metrics if app_state.service else None
    return {"name": (metrics or {}).get("model", "not trained"), "metrics": metrics or {}}
