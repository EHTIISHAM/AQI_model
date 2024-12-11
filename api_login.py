import requests
from dotenv import load_dotenv, set_key
import os

# Load the .env file or create one if it doesn't exist
env_file = ".env"
if not os.path.exists(env_file):
    with open(env_file, "w") as f:
        f.write("")

load_dotenv(env_file)

# API details
url = "https://airquality.aqi.in/api/v1/login"
headers = {"Content-Type": "application/x-www-form-urlencoded"}
data = {
    "email": "ehtasham7899@gmail.com",
    "password": "1Xc+VT141M0q"
}

# Send the POST request
response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    response_json = response.json()
    token = response_json.get("token")

    if token:
        # Save the token to the .env file
        set_key(env_file, "AUTH_TOKEN", token)
        print("Token saved to .env file.")
    else:
        print("Error: Token not found in the response.")
else:
    print(f"Error: Request failed with status code {response.status_code}")
    print(response.text)