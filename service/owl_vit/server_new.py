import base64
import io
import torch
import uvicorn
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image

# Load OWL-ViT model and processor
ckpt_name = "google/owlv2-large-patch14-ensemble"
processor = Owlv2Processor.from_pretrained(ckpt_name)
model = Owlv2ForObjectDetection.from_pretrained(ckpt_name).to("cuda")

app = FastAPI(title="OWL-ViT Object Detection API", version="1.0")

# Request Models
class ObjectDetectionRequest(BaseModel):
    text_queries: List[str]
    image: str  # Base64 encoded image
    bbox_conf_threshold: float = 0.2

class ImageMatchRequest(BaseModel):
    image: str  # Base64 encoded image
    query_image: str  # Base64 encoded reference image
    match_threshold: float = 0.2
    nms_threshold: float = 1.0

# Utility functions
def decode_base64(base64_image: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_image)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def predict_boxes(image: Image.Image, text_queries: List[str], bbox_conf: float) -> dict:
    inputs = processor(text=[text_queries], images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    padded_image_size = inputs.pixel_values.shape[2:]  # (height, width)
    target_sizes = torch.Tensor([padded_image_size]).cuda()
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=bbox_conf)[0]
    
    bboxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    box_names = [text_queries[i] for i in labels]
    return {"box_names": box_names, "bboxes": bboxes.tolist(), "scores": scores.tolist()}

def match_by_image(image: Image.Image, query_image: Image.Image, match_threshold: float, nms_threshold: float) -> dict:
    inputs = processor(images=image, query_images=query_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
    
    target_sizes = torch.Tensor([inputs.pixel_values.shape[2:]]).cuda()
    results = processor.post_process_image_guided_detection(outputs, threshold=match_threshold, nms_threshold=nms_threshold, target_sizes=target_sizes)[0]
    return {"bboxes": results["boxes"].tolist(), "scores": results["scores"].tolist()}

# FastAPI Endpoints
@app.post("/owl_detect", summary="Detect objects in an image based on text queries")
async def api_detect_objects(request: ObjectDetectionRequest):
    image = decode_base64(request.image)
    results = predict_boxes(image, request.text_queries, request.bbox_conf_threshold)
    return results

@app.post("/owl_match_by_image", summary="Match objects in an image using a reference image")
async def api_match_by_image(request: ImageMatchRequest):
    image = decode_base64(request.image)
    query_image = decode_base64(request.query_image)
    results = match_by_image(image, query_image, request.match_threshold, request.nms_threshold)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
