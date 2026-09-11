from fastapi import APIRouter


router = APIRouter()


# --------------------------------------------------
# Health check
# --------------------------------------------------

@router.get("")
def health():
    return {
        "status": "healthy"
    }