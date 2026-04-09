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
    if runner.bench_running:
        raise HTTPException(
            409, "A benchmark is currently running. Chat is paused until it finishes."
        )

    async def event_stream():
        error_msg: str | None = None
        try:
            async for event in runner.stream_chat(
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                if event["type"] == "delta":
                    yield f"data: {json.dumps({'delta': event['text']})}\n\n"
                elif event["type"] == "usage":
                    payload = {
                        "usage": {
                            "prompt_tokens": event["prompt_tokens"],
                            "completion_tokens": event["completion_tokens"],
                        }
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            error_msg = str(e)
        if error_msg:
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
