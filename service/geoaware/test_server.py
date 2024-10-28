import requests
import base64
import time
# Load and encode image
with open('../../images/2.jpg', 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

start_time = time.time()

# Send request
response = requests.post('http://crane1.d2.comp.nus.edu.sg:55575/process_image', 
                        json={'image': img_base64})
features = response.json()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# print(features['features'])
print(features['shape'])
# print(features)
