from PIL import Image
from apis.language_model import LLaVA

client = LLaVA()

"""Test the LLaVA chat function with an image and a prompt."""
prompt = "where is the fruits?"
image = Image.open("images/3.jpg")  # Adjust path as necessary
response = client.chat(prompt, image)
# assert isinstance(response, str)  # Ensure the response is a string

print("\nLLaVA responded:\n", response, '\n', '_'*50)

