from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Anchor API")


class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class GenerateResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # placeholder — real model loading comes later
    return GenerateResponse(
        status="stub",
        message=f"Received prompt: '{req.prompt}' (model not loaded yet)",
    )