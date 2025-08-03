from fastapi import APIRouter, HTTPException, Depends
from models.request_models import ReplyRequest, ReplyResponse
from service.reply_generation.reply_service import ReplyService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/shatiku-ai", tags=["reply"])


def get_reply_service() -> ReplyService:
    return ReplyService()


@router.post("/generate-reply", response_model=ReplyResponse)
async def generate_reply(
    request: ReplyRequest,
    reply_service: ReplyService = Depends(get_reply_service)
) -> ReplyResponse:
    try:
        logger.info(f"自動返信リクエストを受信: ユーザー {request.settings.userId}, チャンネル {request.settings.channel}")
        
        result = await reply_service.generate_reply(
            request=request,
            max_length=512,
            temperature=0.7,
            top_p=0.9
        )
        
        response = ReplyResponse(
            reply=result["reply"],
            replyAt=result["replyAt"]
        )
        
        logger.info(f"自動返信を生成: {result['reply'][:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"自動返信生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"自動返信の生成に失敗しました: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "reply_generation"}