import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'dir_average':2, 'act1_average':9, 'act2_average':6})

print(r.json())