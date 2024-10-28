from transformers import Pipeline
from tqdm import tqdm
import torch
import numpy as np

from pprint import pp
from PIL import Image
import os
import time
import requests
import json
import gradio as gr
import transformers
import torch
import tempfile
import numpy as np
from pipeline import TiOPipeline

model_id = "jxu124/SInViG-240117-1-to-2"
model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True).half().cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
image_processor = transformers.AutoImageProcessor.from_pretrained(model_id)

tio_pipeline = TiOPipeline(
    model=model, tokenizer=tokenizer, image_processor=image_processor, device="cuda", torch_dtype=torch.float16)

image = Image.open("../../images/1.png")
bbox = [0.161, 0.661, 0.314, 0.775]
dialog = [image, "Can you help me with?"]
tio_pipeline({"chatbot": [dialog]})[0]
# print(gens)

# \
# from tools import show_mask
# show_mask(image, bbox_pred)