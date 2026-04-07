import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest
from app.services.vllm_runner import runner

router = APIRouter(tags=["inference"])


@router.post("/chat")
async def chat(req: ChatRequest):
    if not runner.loaded:
        raise HTTPException(409, "No model loaded. POST /models/load first.")

    async def event_stream():
        try:
            async for delta in runner.stream_chat(
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
