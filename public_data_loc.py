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
base_url = "https://airquality.aqi.in/api/v1/GetPublicstationHistoricDataByLocationId"

class uniquestr(str):

    _lower = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def lower(self):
        if self._lower is None:
            lower = str.lower(self)
            if str.__eq__(lower, self): 
                self._lower = self
            else:
                self._lower = uniquestr(lower)
        return self._lower

        
# Function to fetch data with pagination
def fetch_public_station_data(loc_id):
    authorization = f"bearer {auth_token}"
    loc_id = str(loc_id)
    headers = {
        uniquestr("Authorization"): authorization,
        uniquestr("locationid"): loc_id
        }
    #params = {"locationid ": loc_id}
    print(type(headers))
    # Send the GET request
    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"Data fetched for {loc_id}. Total records: {len(data)}")
        return data
    else:
        print(f"Error: Failed to fetch data of {loc_id}. Status code: {response.status_code}")
        print(response.text)
        return None

# Main logic to fetch paginated data
loc_id = 359

data = fetch_public_station_data(loc_id=loc_id)
if data==None:
    print("No Data Exiting")
    exit()
print(f"Total records fetched: {len(data)}")

df_data = {'loc_id':[],'time_stamp':[], 'no2':[], 'w':[], 't':[],  'AQI-IN':[], 'pm10':[], 'aqi':[], 'co':[], 'p':[],  'pm25':[], 'wg':[], 'h':[], 'o3':[]}
print(df_data)
sensorname_all = []

"""for loc in data["Locations"]:
    for past_data in loc["pastdata"]:
        for single_data in past_data:
            if single_data["sensorname"] == 'no2':
                df_data['time_stamp'].append(single_data["created_at"])
                df_data['loc_id'].append(single_data["uid"])
"""

for loc in data["Locations"]:
    for past_data in loc["pastdata"]:
        for single_data in past_data:
            if single_data["created_at"] not in df_data["time_stamp"]:
                # Add a new row with timestamp and placeholders
                df_data["time_stamp"].append(single_data["created_at"])
                for key in df_data.keys():
                    if key != "time_stamp":
                        df_data[key].append(None)
            idx = df_data["time_stamp"].index(single_data["created_at"])
            if single_data["sensorname"] in df_data:
                df_data[single_data["sensorname"]][idx] = single_data["sensorvalue"]
            #print(single_data["sensorname"],single_data["sensorvalue"],single_data["created_at"])
print(len(df_data["loc_id"]))
df = pd.DataFrame(df_data)
df['loc_id'] = df['loc_id'].fillna(loc_id)
df.to_csv("sample_data.csv")
print(df.head())