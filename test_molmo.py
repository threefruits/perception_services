from PIL import Image
import requests
import base64
from io import BytesIO

def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def chat(prompt, image, meta_prompt=""):
    base64_image = convert_pil_image_to_base64(image)
    payload = {
        "prompt": meta_prompt + '\n' + prompt,
        "image": base64_image,
        "max_new_tokens": 2048,
        "temperature": 0.0
    }
    response = requests.post(
        "http://crane5.d2.comp.nus.edu.sg:4007/molmo_chat", 
        json=payload,
    ).json()
    return response["text"]


if __name__ == "__main__":
    image = Image.open("images/3.jpg")
    prompt = "point to the orange"
    response = chat(prompt, image)
    print(response)
    