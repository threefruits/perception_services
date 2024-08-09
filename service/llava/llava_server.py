from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import argparse

parser = argparse.ArgumentParser(description='LLaVA Server')
parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
parser.add_argument('--port', default=55576, type=int, help='Port number to run the app on')
parser.add_argument('--model_id', default='llava-hf/llava-v1.6-mistral-7b-hf', type=str, help='Model ID to use for inference')
args = parser.parse_args()

processor = LlavaNextProcessor.from_pretrained(args.model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True, attn_implementation="flash_attention_2") 

# Flask app
app = Flask(__name__)


def convert_base64_to_pil_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return image


@app.route('/llava_chat', methods=['POST'])
def llava_chat():
    # Parse JSON data
    query_data = request.get_json()
    base64_image = query_data["image"]
    prompt = query_data["prompt"]
    temperature = query_data["temperature"]
    max_new_tokens = query_data.get("max_new_tokens", 1024)

    # Convert base64 to PIL Image
    image = convert_base64_to_pil_image(base64_image)
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, temperature=temperature, max_new_tokens=100)
    text_outputs = processor.decode(output[0], skip_special_tokens=True)

    # Return results as JSON
    return jsonify({
        'text': text_outputs
    })

if __name__ == '__main__':


    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)
