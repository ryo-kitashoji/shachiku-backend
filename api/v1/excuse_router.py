from fastapi import APIRouter, HTTPException, Depends
from models.request_models import ExcuseRequest, ExcuseResponse
from service.excuse_generation.excuse_service import ExcuseService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/excuse", tags=["excuse"])


def get_excuse_service() -> ExcuseService:
    return ExcuseService()


@router.post("/generate", response_model=ExcuseResponse)
async def generate_excuse(
    request: ExcuseRequest,
    excuse_service: ExcuseService = Depends(get_excuse_service)
) -> ExcuseResponse:
    try:
        logger.info(f"質問を受信: {request.question}")
        
        excuse = await excuse_service.generate_excuse(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        response = ExcuseResponse(
            question=request.question,
            excuse=excuse["text"],
            confidence=excuse["confidence"]
        )
        
        logger.info(f"言い訳を生成: {excuse['text'][:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"言い訳生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"言い訳の生成に失敗しました: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "excuse_generation"}