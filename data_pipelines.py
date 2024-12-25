import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Create groups based on location and time
def find_near_stations_knn(df, lat, lon, target_loc_id, k=5):
    """
    Find the k nearest stations to a given latitude and longitude, excluding the target location.
    """
    spatial_features = df[["lat", "lon"]].values  # Extract spatial features
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')  # Add 1 to ensure we can exclude the target location
    knn.fit(spatial_features)
    distances, indices = knn.kneighbors([[lat, lon]])

    # Map indices to loc_id
    nearest_loc_ids = df.iloc[indices.flatten()]["loc_id"].values

    # Exclude the target location
    nearest_loc_ids = [loc_id for loc_id in nearest_loc_ids if loc_id != target_loc_id]
    
    # Return only k neighbors
    return nearest_loc_ids[:k]

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
    filtered_df = filtered_df[filtered_df["loc_id"] != target_loc_id]
    
    # Calculate Euclidean distance for more accurate filtering
    filtered_df["distance"] = np.sqrt(filtered_df["lat_diff"]**2 + filtered_df["lon_diff"]**2)
    
    # Sort by distance
    filtered_df = filtered_df.sort_values(by="distance")
    
    # If k is specified, return the top-k closest stations
    if k is not None:
        filtered_df = filtered_df.head(k)
    
    # Return loc_id of the nearby stations
    return filtered_df["loc_id"].values


def normalize_data_with_range(df, columns, ranges):
    """
    Normalize pollutant data using specified min/max ranges.
    
    Parameters:
    - df: DataFrame containing pollutant data.
    - columns: List of column names to normalize.
    - ranges: Dictionary of column names to (min, max) tuples.
    
    Returns:
    - Normalized DataFrame.
    """
    for col in columns:
        if col in ranges:
            min_val, max_val = ranges[col]
            df[col] = (df[col] - min_val) / (max_val - min_val)
            # Clip values to ensure they remain within [0, 1]
            df[col] = df[col].clip(0, 1)
        else:
            print(f"Warning: {col} not found in ranges. Skipping normalization.")
    return df

pollutant_ranges = {
    "no2": (0, 500),
    "w": (0, 76.84),
    "t": (-100, 50.9),
    "AQI-IN": (0, 500),
    "pm10": (1, 510),
    "aqi": (0, 300),
    "co": (0, 50000),
    "p": (940, 1046),
    "pm25": (1, 380),
    "wg": (0, 140),
    "h": (0, 90),
    "o3": (0, 1250),
}

def create_location_neighbors(df, max_distance=5, k=None):
    """
    Create a dictionary where each loc_id maps to its nearby station loc_ids.
    """
    loc_neighbors = {}

    # Extract unique location ids and their attributes
    unique_locations = df.drop_duplicates(subset=["loc_id"])[["loc_id", "lat", "lon"]]

    for _, loc in unique_locations.iterrows():
        loc_id = loc["loc_id"]
        lat, lon = loc["lat"], loc["lon"]

        # Find nearby locations using the manual function
        neighbors = find_near_stations_manual(df, lat, lon, loc_id, max_distance, k)

        # Map loc_id to its neighbors
        loc_neighbors[loc_id] = neighbors.tolist()

    return loc_neighbors

def generate_hourly_df(df, location_neighbors_dict):
    """
    Generate a DataFrame for each loc_id using its nearby locations, grouped by hour.
    Ensures each hour has exactly 6 rows by filling with zeros or flags if exceeded.
    """
    hourly_dfs = {}

    for loc_id, neighbors in location_neighbors_dict.items():
        # Filter data for the current loc_id and its neighbors
        filtered_df = df[df["loc_id"].isin([loc_id] + neighbors)].copy()

        # Convert timestamps to datetime for easy grouping
        filtered_df["time_stamp"] = pd.to_datetime(filtered_df["time_stamp"])
        filtered_df["hour"] = filtered_df["time_stamp"].dt.floor("h")  # Group by hour

        # Group by hour
        grouped = filtered_df.groupby("hour")

        # Process each group
        hourly_data = []
        for hour, group in grouped:
            if len(group) < 6:
                # Fill with zeros if the count is less than 6
                missing_count = 6 - len(group)
                zeros_df = pd.DataFrame({
                    col: [0] * missing_count for col in filtered_df.columns if col != "time_stamp"
                })
                zeros_df["time_stamp"] = [hour] * missing_count
                group = pd.concat([group, zeros_df], ignore_index=True)
            elif len(group) > 6:
                print(f"Exceeded rows for hour {hour} in loc_id {loc_id}")
                continue  # Skip this hour

            # Append processed group
            hourly_data.append(group)

        # Combine all hourly data into a single DataFrame
        if hourly_data:
            hourly_dfs[loc_id] = pd.concat(hourly_data, ignore_index=True)
    for loc_id in hourly_dfs:
        hourly_dfs[loc_id] = hourly_dfs[loc_id].fillna(0)

    return hourly_dfs


# Main pipeline function
def pipeline_demo(path_to_csv):
    """
    Demonstrate the pipeline with simulated data.
    """
    # Generate random data
    raw_data = pd.read_csv(path_to_csv)
    if 'Unnamed: 0' in raw_data.columns:
        print("found Unamed: 0 in raw data removing it")
        predictors_df = raw_data.drop(columns=['Unnamed: 0'])

    # Normalize pollutant data
    pollutant_columns = ["no2", "w", "t", "AQI-IN", "pm10", "aqi", "co", "p", "pm25", "wg", "h", "o3"]
    normalized_data = normalize_data_with_range(raw_data.copy(), pollutant_columns,pollutant_ranges)
    # Example group creation for a random station and timestamp
    loc_dict = create_location_neighbors(normalized_data,k=5)
    #print(loc_dict.keys())
    hourly_dfs = generate_hourly_df(normalized_data,loc_dict)
    #print(hourly_dfs.get(364))
    output_path = "loc_336_hourly_data.csv"
    #hourly_dfs[336].to_csv(output_path, index=False)
    # Process hourly DataFrames to generate predictors (X) and targets (Y)
    X = []  # Predictors
    Y = []  # Targets

    for loc_id, hourly_data in hourly_dfs.items():
        # Group the data by the hour (assuming `hour` is preprocessed; otherwise group by `time_stamp`)
        hourly_groups = hourly_data.groupby("time_stamp")
        
        for hour, hour_df in hourly_groups:
            # Ensure `hour_df` is structured correctly
            if hour_df.empty:
                print(f"Skipping loc_id {loc_id} for hour {hour} due to empty DataFrame")
                continue

            # Find the target row for the given `loc_id`
            target_row_index = hour_df.index[hour_df["loc_id"] == loc_id].tolist()
            if not target_row_index:
                #print(f"Skipping hour {hour} for loc_id {loc_id} (target row not found)")
                continue

            # Target row is the one matching `loc_id`
            target_row = hour_df.loc[target_row_index[0]]

            # Drop `time_stamp` from both predictors and target
            hour_df = hour_df.drop(columns=["time_stamp"])

            # Predictors are all rows excluding the target row
            predictors_df = hour_df.drop(index=target_row_index).drop(columns=["lat", "lon", "elevation", "loc_id","hour"])

            # Replace NaN values with 0 in predictors and target
            #predictors_df = predictors_df.fillna(0)
            #target_row = target_row.fillna(0)

            # Target vector contains only pollutant values for the target row
            target_values = target_row.drop(["lat", "lon", "elevation", "loc_id","time_stamp","hour"]).values.astype(float)

            # Validate the predictors DataFrame shape
            if predictors_df.shape[1] != len(target_values):  # Ensure feature count matches
                print(f"Warning: Skipping hour {hour} for loc_id {loc_id} due to invalid X shape")
                continue

            # Validate the number of rows for predictors
            if predictors_df.shape[0] != 5:  # Ensure we have data for 5 stations
                #print(f"Warning: Skipping hour {hour} for loc_id {loc_id} due to insufficient stations in hourly data")
                continue
            if 'Unnamed: 0' in predictors_df.columns:
                print("found Unamed: 0 removing it")
                predictors_df = predictors_df.drop(columns=['Unnamed: 0'])
            # Append to X and Y
            X.append(predictors_df.values.flatten())  # Flatten predictors to match the required shape
            Y.append(target_values)
    # Convert X and Y to numpy arrays for model training
    X = np.array(X)
    Y = np.array(Y)

    print(f"Processed {len(X)} samples for X and {len(Y)} samples for Y")
    
    return X, Y

if __name__ == '__main__':
    X, Y = pipeline_demo("combined_data.csv")
    np.save("X.npy",X)
    np.save("Y.npy",Y)
    