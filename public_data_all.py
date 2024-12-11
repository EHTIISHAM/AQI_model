import requests
from dotenv import load_dotenv
import os
import pandas as pd

# Load the token from the .env file
load_dotenv(".env")
auth_token = os.getenv("AUTH_TOKEN")

if not auth_token:
    raise ValueError("Authorization token not found in .env file. Please log in first.")

# Base URL for the API
base_url = "https://airquality.aqi.in/api/v1/GetPublicstationHistoricData"

# Function to fetch data with pagination
def fetch_public_station_data(skip=0, take=500):
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"skip": skip, "take": take}

    # Send the GET request
    response = requests.get(base_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"Data fetched for skip={skip}, take={take}. Total records: {len(data)}")
        return data
    else:
        print(f"Error: Failed to fetch data for skip={skip}, take={take}. Status code: {response.status_code}")
        print(response.text)
        return None

# Main logic to fetch paginated data
skip = 0
take = 2

data = fetch_public_station_data(skip=skip, take=take)

if data==None:
    print("No Data Exiting")
    exit()
print(f"Total records fetched: {len(data)}")

for loc in data["Locations"]:
    print(loc["stationname"])
    print(loc["locationId"])