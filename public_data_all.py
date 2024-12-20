import requests
import pandas as pd
import os
import pandas as pd

# Load the token from the .env file
auth_token = os.environ.get("AUTH_TOKEN")

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

all_step_dfs = []

skip = 0
take = 1500

data = fetch_public_station_data(skip=skip, take=take)

if data==None:
    print("No Data Exiting")
    exit()
print(f"Total records fetched: {len(data)}")

df_data = {'loc_id':[],'lat':[],'lon':[],'elevation':[],'time_stamp':[], 'no2':[], 'w':[], 't':[],  'AQI-IN':[], 'pm10':[], 'aqi':[], 'co':[], 'p':[],  'pm25':[], 'wg':[], 'h':[], 'o3':[]}
all_loc_df = {'main_loc_id':[]}
for loc in data["Locations"]:
    print(loc['lat'],loc['lon'],loc['Elevation'])
    #print(loc["locationId"])



# Initialize a dictionary to hold data for all locations
all_data_frames = []

# Loop through each location in the data
for loc in data["Locations"]:
    lat = loc["lat"]
    lon = loc["lon"]
    eleva = loc["Elevation"]
    locationid = loc["locationId"]

    # Initialize a temporary dictionary for the current location
    loc_data = {
        "loc_id": [],
        "lat": [],
        "lon": [],
        "elevation": [],
        "time_stamp": [],
        "no2": [],
        "w": [],
        "t": [],
        "AQI-IN": [],
        "pm10": [],
        "aqi": [],
        "co": [],
        "p": [],
        "pm25": [],
        "wg": [],
        "h": [],
        "o3": [],
    }

    # Process past data for the current location
    for past_data in loc["pastdata"]:
        for single_data in past_data:
            created_at = single_data["created_at"]

            # Check if timestamp is already present
            if created_at not in loc_data["time_stamp"]:
                # Add a new row with timestamp and placeholders
                loc_data["time_stamp"].append(created_at)
                for key in loc_data.keys():
                    if key not in ["time_stamp"]:
                        loc_data[key].append(None)

            # Find the index of the current timestamp
            idx = loc_data["time_stamp"].index(created_at)

            # Update sensor data if applicable
            sensorname = single_data["sensorname"]
            if sensorname in loc_data:
                loc_data[sensorname][idx] = single_data["sensorvalue"]

            # Populate static location details
            if not loc_data["lat"][idx] == lat:
                loc_data["lat"][idx] = lat
                loc_data["lon"][idx] = lon
                loc_data["elevation"][idx] = eleva
                loc_data["loc_id"][idx] = locationid

    # Convert the location data to a DataFrame and append it to the list
    loc_df = pd.DataFrame(loc_data)
    all_data_frames.append(loc_df)

# Combine all DataFrames into one
final_df = pd.concat(all_data_frames, ignore_index=True)
final_df.to_csv('combined_data.csv',index=False)