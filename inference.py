import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import requests
import pandas as pd
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_pipelines import pollutant_ranges, normalize_data_with_range

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


def find_near_stations_manual(df, lat, lon, target_loc_id, max_distance=3, k=None):
    """
    Find nearby stations within a maximum distance (in units of lat/lon), excluding the target location.
    Optionally return the top-k closest stations.
    """
    # Extract unique locations based on loc_id
    unique_locations = df[["loc_id", "lat", "lon"]].drop_duplicates(subset=["loc_id"])

    # Calculate the absolute difference in latitudes and longitudes
    unique_locations["lat_diff"] = np.abs(unique_locations["lat"] - lat)
    unique_locations["lon_diff"] = np.abs(unique_locations["lon"] - lon)
    
    # Filter rows where lat_diff and lon_diff are within the max_distance
    filtered_df = unique_locations[
        (unique_locations["lat_diff"] <= max_distance) & (unique_locations["lon_diff"] <= max_distance)
    ]

    # Exclude the target location
    filtered_df = filtered_df[(filtered_df["lat"] != lat) & (filtered_df["lon"] != lon)]
    
    # Calculate Euclidean distance for more accurate filtering
    filtered_df["distance"] = np.sqrt(filtered_df["lat_diff"]**2 + filtered_df["lon_diff"]**2)
    
    # Sort by distance
    filtered_df = filtered_df.sort_values(by="distance")
    
    # If k is specified, return the top-k closest stations
    if k is not None:
        filtered_df = filtered_df.head(k)
    
    # Return loc_id of the nearby stations
    return filtered_df["loc_id"].values


def create_location_neighbors(df,locations_list, max_distance=3, k=5):
    """
    Create a dictionary where each loc_id (index in the list) maps to its nearby station loc_ids.
    """
    loc_neighbors = {}

    # Iterate through each location in the list
    for target_loc_id, predictor_data in enumerate(locations_list):
        # Find neighbors for the current location
        neighbors = find_near_stations_manual(df,float( predictor_data[1]), float(predictor_data[2]), target_loc_id, max_distance, k)

        # Map loc_id to its neighbors
        loc_neighbors[predictor_data[0]] = neighbors.tolist()

    return loc_neighbors

def denormalize_data(values, ranges):
    return values * (ranges[1] - ranges[0]) + ranges[0]


all_step_dfs = []

import pandas as pd
from datetime import datetime, timedelta

# Initialize the DataFrame schema
df_data = {
    'loc_id': [],
    'lat': [],
    'lon': [],
    'elevation': [],
    'time_stamp': [],
    'no2': [],
    'w': [],
    't': [],
    'AQI-IN': [],
    'pm10': [],
    'aqi': [],
    'co': [],
    'p': [],
    'pm25': [],
    'wg': [],
    'h': [],
    'o3': []
}

# Fetch data
skip = 0
take = 1500
data = fetch_public_station_data(skip=skip, take=take)

if data is None:
    print("No Data Exiting")
    exit()

print(f"Total records fetched: {len(data)}")

# Process each location
for loc in data["Locations"]:
    loc_id = loc["locationId"]
    lat = loc['lat']
    lon = loc['lon']
    elevation = loc['Elevation']
    last_updated = loc['last_updated']

    # Extract and process sensor data
    sensor_data = {  # Default all sensors to 0 in case of missing data
        'no2': 0, 'w': 0, 't': 0, 'AQI-IN': 0, 'pm10': 0, 'aqi': 0, 'co': 0,
        'p': 0, 'pm25': 0, 'wg': 0, 'h': 0, 'o3': 0
    }

    for sub_data in loc['airComponents']:
        sensor_name = sub_data['sensorName']
        sensor_value = sub_data['sensorData']
        if sensor_name in sensor_data:
            sensor_data[sensor_name] = sensor_value

    # Normalize the timestamp
    timestamp = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
    rounded_hour = timestamp.replace(minute=0, second=0, microsecond=0)

    # Append processed data
    df_data['loc_id'].append(loc_id)
    df_data['lat'].append(float(lat))
    df_data['lon'].append(float(lon))
    df_data['elevation'].append(float(elevation))
    df_data['time_stamp'].append(rounded_hour)
    df_data['no2'].append(float(sensor_data['no2']))
    df_data['w'].append(float(sensor_data['w']))
    df_data['t'].append(float(sensor_data['t']))
    df_data['AQI-IN'].append(float(sensor_data['AQI-IN']))
    df_data['pm10'].append(float(sensor_data['pm10']))
    df_data['aqi'].append(float(sensor_data['aqi']))
    df_data['co'].append(float(sensor_data['co']))
    df_data['p'].append(float(sensor_data['p']))
    df_data['pm25'].append(float(sensor_data['pm25']))
    df_data['wg'].append(float(sensor_data['wg']))
    df_data['h'].append(float(sensor_data['h']))
    df_data['o3'].append(float(sensor_data['o3']))


# Convert to DataFrame
pollutant_columns = ["no2", "w", "t", "AQI-IN", "pm10", "aqi", "co", "p", "pm25", "wg", "h", "o3"]

sensor_df = pd.DataFrame(df_data)
sensor_df = normalize_data_with_range(sensor_df, pollutant_columns, pollutant_ranges)

# Display summary
print("Processed DataFrame:")
print(sensor_df.head())

predictor_df = pd.read_csv('Stations.csv')
predictor_df = predictor_df.drop(["stationname","locationName","cityname","statename","country","timezoneName"],axis=1)

predictor_list = predictor_df[["uid","lat","lon","Elevation"]].values.tolist()
#print(predictor_list[:5])
near_location = create_location_neighbors(df=sensor_df,locations_list=predictor_list)


###
# from here prediction
###

model = tf.keras.models.load_model("best_model.keras")

# DataFrames: sensor_df is your real-time data, df_results will store results
df_results = pd.DataFrame(columns=['loc_id', 'time_stamp'] + list(pollutant_ranges.keys()))

# Process each main location
for main_loc_id, near_loc_ids in near_location.items():
    # Fetch main loc_id data
    main_data = sensor_df[sensor_df["loc_id"] == main_loc_id]

    if len(near_loc_ids) < 5:
        # If less than 5 neighbors, append zeros
        result_row = {key: 0 for key in pollutant_ranges.keys()}
        result_row['loc_id'] = main_loc_id
        result_row['time_stamp'] = main_data['time_stamp'].iloc[0] if not main_data.empty else None
        result_df = pd.DataFrame([result_row])  # Convert the row into a DataFrame
        df_results = pd.concat([df_results, result_df], ignore_index=True)
        continue

    # Fetch near_loc_id data
    near_data = sensor_df[sensor_df["loc_id"].isin(near_loc_ids)]

    # If near_loc_id data is insufficient, append zeros
    if len(near_data) < 5:
        result_row = {key: 0 for key in pollutant_ranges.keys()}
        result_row['loc_id'] = main_loc_id
        result_row['time_stamp'] = main_data['time_stamp'].iloc[0] if not main_data.empty else None
        result_df = pd.DataFrame([result_row])  # Convert the row into a DataFrame
        df_results = pd.concat([df_results, result_df], ignore_index=True)
        continue

    # Prepare input for the model (flatten values for Y)
    Y = near_data[list(pollutant_ranges.keys())].iloc[:5].values
    if len(Y) >60:
        print("len_decreased")
        Y = Y[:60]  # Ensure fixed-size input (5 neighbors x 12 pollutants = 60 features)
    Y = np.expand_dims(Y, axis=0)
    # Predict pollutants
    predicted_values = model.predict(Y)  # Reshape for a single prediction
    predicted_values = predicted_values.flatten()

    # Denormalize predicted values
    denormalized_values = [
        denormalize_data(predicted_values[i], pollutant_ranges[col])
        for i, col in enumerate(pollutant_ranges.keys())
    ]

    # Append results to df_results
    result_row = {col: denormalized_values[i] for i, col in enumerate(pollutant_ranges.keys())}
    result_row['loc_id'] = main_loc_id
    result_row['time_stamp'] = main_data['time_stamp'].iloc[0] if not main_data.empty else None
    result_df = pd.DataFrame([result_row])  # Convert the row into a DataFrame
    df_results = pd.concat([df_results, result_df], ignore_index=True)


# Final Results
print("Processed Results:")
print(df_results.head())
df_results.to_csv('station_live_prediction.csv')
