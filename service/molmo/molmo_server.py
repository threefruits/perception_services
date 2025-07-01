from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import argparse

parser = argparse.ArgumentParser(description='Molmo Server')
parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
parser.add_argument('--port', default=4007, type=int, help='Port number to run the app on')
parser.add_argument('--model_id', default='allenai/Molmo-7B-D-0924', type=str, help='Model ID to use for inference')
parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit mode')
# parser.add_argument('--model_path', default=None, type=str, help='Model path to use for inference')

args = parser.parse_args()

model_id = args.model_id

# load the processor
processor = AutoProcessor.from_pretrained(
    args.model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

if args.load_in_4bit:
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
        quantization_config=bnb_config
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
    )
    
# Flask app
app = Flask(__name__)

def convert_base64_to_pil_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return image

@app.route('/molmo_chat', methods=['POST'])
def llava_chat():
    # Parse JSON data
    query_data = request.get_json()
    base64_image = query_data["image"]
    prompt = query_data["prompt"]
    temperature = query_data["temperature"]
    max_new_tokens = query_data.get("max_new_tokens", 1024)
    # Convert base64 to PIL Image
    image = convert_base64_to_pil_image(base64_image)

    inputs = processor.process(
        images=[image],
        text=prompt
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Return results as JSON
    return jsonify({
        'text': generated_text
    })

if __name__ == '__main__':

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)
