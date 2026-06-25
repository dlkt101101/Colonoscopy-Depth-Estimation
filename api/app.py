from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import numpy as np
from src.inference import run_inference

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/depth")
def predict_depth(file: UploadFile = File(...)):
    image = file.read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    depth_map = run_inference(image)

    normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_image = (normalized * 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_image)

    buf = io.BytesIO()
    depth_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")