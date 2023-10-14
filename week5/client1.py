import requests

url = "http://127.0.0.1:9696/predict"
client = {"job": "retired", "duration": 445, "poutcome": "success"}
res = requests.post(url, json=client).json()

print(res)
