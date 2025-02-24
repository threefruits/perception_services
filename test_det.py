import os
import sys
import requests
# Root folder
sys.path.append("../..")

from PIL import Image

from apis.owlv2 import OWLViT, visualize_image
detector = OWLViT(server_url="http://crane5.d2.comp.nus.edu.sg:4000")

import time
# file_path = "../test.jpg"
# image = Image.open(file_path)
url = "https://www.californiastrawberries.com/wp-content/uploads/2021/05/Rainbow-Fruit-Salad-1024-500x375.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# Example text queries
text_queries = [ 'apple']

time1 = time.time()

det_data = detector.detect_objects(
    image=image,
    text_queries=text_queries,
    bbox_score_top_k=20,
    bbox_conf_threshold=0.12
)

time2 = time.time()
print(f"Time taken: {time2 - time1:.2f}s")

for i, item in enumerate(det_data):
    print(f"--- Detection {i + 1} ---")
    print(f"Box Name: {item['box_name']}")
    print(f"Score: {item['score']}")
    print(f"Box Coordinates: {item['bbox']}")