import io
import base64
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import time

from model.classifier import CIFAR10Classifier, CIFAR10_CLASSES
from model.gradcam import GradCAM

app = FastAPI(
    title="CIFAR-10 视觉模型可解释性沙盒",
    description="基于 Grad-CAM 的实时图像分类与注意力可视化",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

classifier: Optional[CIFAR10Classifier] = None
gradcam: Optional[GradCAM] = None

class ClassificationResponse(BaseModel):
    success: bool
    classes: List[str]
    probabilities: List[float]
    original_image: str
    heatmap: List[List[float]]
    inference_time: float

class ThresholdRequest(BaseModel):
    threshold: float
    heatmap: List[List[float]]
    original_image: str

class ThresholdResponse(BaseModel):
    success: bool
    overlay_image: str

def encode_image_to_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def decode_image_from_base64(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

@app.on_event("startup")
async def load_model():
    global classifier, gradcam
    print("正在加载 CIFAR-10 分类模型...")
    classifier = CIFAR10Classifier(pretrained=True)
    gradcam = GradCAM(classifier)
    print("模型加载完成，服务已就绪。")

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>页面未找到</h1>")

@app.get("/api/classes")
async def get_classes():
    return {"classes": CIFAR10_CLASSES}

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    if classifier is None or gradcam is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")
    
    try:
        start_time = time.time()
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        classes, probabilities, _ = classifier.predict(image)
        
        original, overlay, heatmap_full = gradcam.generate_overlay(image, threshold=0.5)
        
        original_base64 = encode_image_to_base64(original)
        heatmap_list = heatmap_full.tolist()
        
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        return ClassificationResponse(
            success=True,
            classes=classes,
            probabilities=probabilities,
            original_image=original_base64,
            heatmap=heatmap_list,
            inference_time=inference_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像处理失败: {str(e)}")

@app.post("/api/apply-threshold", response_model=ThresholdResponse)
async def apply_threshold(request: ThresholdRequest):
    try:
        threshold = max(0.0, min(1.0, request.threshold))
        heatmap = np.array(request.heatmap)
        original_image = decode_image_from_base64(request.original_image)
        
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        img_array = np.array(original_image)
        original_size = (img_array.shape[1], img_array.shape[0])
        
        heatmap_resized = np.array(heatmap)
        if heatmap_resized.shape != original_size[::-1]:
            heatmap_resized = np.array(Image.fromarray(heatmap_resized).resize(original_size))
        
        heatmap_normalized = heatmap_resized.copy()
        heatmap_normalized[heatmap_normalized < threshold] = 0
        
        import cv2
        heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        mask = heatmap_normalized > threshold
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        overlay = img_array.copy()
        alpha = 0.6
        if mask_3ch.any():
            overlay[mask_3ch] = cv2.addWeighted(
                img_array[mask_3ch].reshape(-1, 3),
                1 - alpha,
                heatmap_color[mask_3ch].reshape(-1, 3),
                alpha,
                0
            ).flatten()
        
        overlay_base64 = encode_image_to_base64(overlay)
        
        return ThresholdResponse(
            success=True,
            overlay_image=overlay_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"阈值处理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
