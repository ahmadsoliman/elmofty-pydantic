from fastapi import APIRouter
from api.schemas.validation import HealthCheckResponse

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {
        "status": "ok",
        "database": True,
        "external_services": {
            "supabase": True,
            "cohere": True
        }
    }
