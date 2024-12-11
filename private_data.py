import requests
from dotenv import load_dotenv
import os
import pandas as pd

# Load the token from the .env file
load_dotenv(".env")
auth_token = os.getenv("AUTH_TOKEN")

if not auth_token:
    raise ValueError("Authorization token not found in .env file. Please log in first.")

# URL for the private user device data
url = "https://airquality.aqi.in/api/v1/GetPrivateUserDevicesList"

# Function to fetch private user device data
def fetch_private_user_device_data():
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Send the GET request
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print("Private user device data fetched successfully.")
        return data
    else:
        print(f"Error: Failed to fetch private user device data. Status code: {response.status_code}")
        print(response.text)
        return None

# Fetch and print the private user device data
device_data = fetch_private_user_device_data()
if device_data:
    #print("Private Device Data:", device_data["data"])
    for data in device_data["data"]:
        print(data["serialNo"])
        for more_data in data["realtime"]:
            print(more_data)

"""for locs in device_data["data"]:
    print(locs['devicename'])"""