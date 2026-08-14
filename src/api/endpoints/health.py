from fastapi import APIRouter
from ..schemas import HealthResponse
router = APIRouter(tags=["system"])
@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from ..main import app_state
    return HealthResponse(status="healthy", service="customer-churn-intelligence", version="2.0.0", model_loaded=app_state.service is not None and app_state.service.pipeline is not None, model_source="local")
