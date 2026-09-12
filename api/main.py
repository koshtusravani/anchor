import base64
import io
import time

from fastapi import FastAPI

from api.model_loader import get_pipeline
from api.schemas import GenerateRequest, GenerateResponse
from db.models import InferenceRequest
from db.session import SessionLocal

app = FastAPI(title="Anchor API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    start_time = time.time()
    status = "success"
    image_base64 = ""

    try:
        pipeline = get_pipeline()
        image = pipeline(
            req.prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception:
        status = "error"
        raise

    finally:
        latency_ms = (time.time() - start_time) * 1000
        session = SessionLocal()
        try:
            session.add(
                InferenceRequest(
                    prompt=req.prompt,
                    model_version="lora_colorful_sneaker_rank16",
                    latency_ms=latency_ms,
                    status=status,
                )
            )
            session.commit()
        finally:
            session.close()

    return GenerateResponse(
        status=status,
        image_base64=image_base64,
        model_version="lora_colorful_sneaker_rank16",
    )