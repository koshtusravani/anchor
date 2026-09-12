from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class GenerateResponse(BaseModel):
    status: str
    image_base64: str
    model_version: str