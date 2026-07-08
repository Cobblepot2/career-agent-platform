import requests

BASE_URL = "http://127.0.0.1:8010"
response = requests.post(f"{BASE_URL}/index/rebuild")
print(response.status_code)
print(response.json())