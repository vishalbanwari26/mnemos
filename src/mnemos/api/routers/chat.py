import uuid

from fastapi import APIRouter

from mnemos.api.deps import ConversationManagerDep
from mnemos.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/users/{user_id}/messages", response_model=ChatResponse)
async def send_message(
    user_id: str, body: ChatRequest, manager: ConversationManagerDep
) -> ChatResponse:
    session_id = body.session_id or uuid.uuid4()
    result = await manager.handle_message(user_id, session_id, body.message)
    return ChatResponse(
        reply=result.reply,
        session_id=session_id,
        memory_used=result.memory_used,
        facts_learned=[f.fact for f in result.facts_learned],
        strategy_used=result.strategy_used,
    )
