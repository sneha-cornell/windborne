import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import pickle
import requests # Import requests for data acquisition
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Import the model

# --- Helper Functions (from the notebook) ---

def find_list_of_candidates(data):
    """Recursively searches for a list that potentially contains balloon data (dicts or lists)."""
    if isinstance(data, list):
        if any(isinstance(item, (dict, list)) for item in data):
             return data
        elif len(data) > 1 and all(isinstance(item, (int, float)) for item in data):
             return data
        else:
            for item in data:
                found = find_list_of_candidates(item)
                if found:
                    return found

    elif isinstance(data, dict):
        for key in ['balloons', 'data', 'positions', 'flights']:
            if key in data and isinstance(data[key], list):
                 if any(isinstance(item, (dict, list)) for item in data[key]):
                    return data[key]
        for value in data.values():
            found = find_list_of_candidates(value)
            if found:
                return found
    return None


def extract_balloon_data(raw_data, hour_index):
    extracted = []
    candidates = find_list_of_candidates(raw_data)

    if not candidates:
        # print(f"Could not find a list of candidates in data from hour {hour_index}. Raw data type: {type(raw_data)}") # Suppress verbose logging
        return extracted

    for i, entry in enumerate(candidates):
        try:
            lat = None
            lon = None
            balloon_id = None
            timestamp = f"{hour_index:02d}H_ago" # Default timestamp

            if isinstance(entry, dict):
                lat = entry.get('lat') or entry.get('latitude')
                lon = entry.get('lon') or entry.get('lng') or entry.get('longitude')
                balloon_id = entry.get('id') or entry.get('balloon_id') or entry.get('name')
                timestamp = entry.get('timestamp') or entry.get('time') or entry.get('ts')

            elif isinstance(entry, list) and len(entry) >= 2:
                lat = entry[0]
                lon = entry[1]
                balloon_id = f"balloon_{hour_index:02d}_{i}"

            if lat is not None and lon is not None:
                try:
                    lat = float(lat)
                    lon = float(lon)
                except (ValueError, TypeError):
                    # print(f"Skipping entry at index {i} from hour {hour_index} due to non-numeric lat/lon: lat={lat}, lon={lon}") # Suppress verbose logging
                    continue

                final_timestamp = timestamp if isinstance(entry, dict) and (entry.get('timestamp') or entry.get('time') or entry.get('ts')) else f"{hour_index:02d}H_ago"

                extracted.append({
                    'id': balloon_id,
                    'lat': lat,
                    'lon': lon,
                    'timestamp': final_timestamp,
                    'raw': entry
                })
            else:
                if isinstance(entry, (dict, list)):
                     pass # Suppress verbose logging
                else:
                     pass # Suppress verbose logging


        except (TypeError, AttributeError, IndexError, Exception) as e:
            # print(f"Error processing entry at index {i} from hour {hour_index}: {entry}. Error: {e}") # Suppress verbose logging
            continue

    return extracted

def fetch_and_extract_all():
    all_balloon_data = []
    st.info("Fetching balloon data...")
    progress_bar = st.progress(0)
    for hour in range(24):
        url = f"https://a.windbornesystems.com/treasure/{hour:02d}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            try:
                data = response.json()
                extracted = extract_balloon_data(data, hour)
                all_balloon_data.extend(extracted)
            except json.JSONDecodeError as e:
                st.warning(f"Malformed JSON at {url}: {e}, skipping.")
            except Exception as e:
                 st.warning(f"Error processing JSON data from {url}: {e}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                st.warning(f"File not found at {url}, skipping.")
            else:
                st.warning(f"HTTP error fetching {url}: {e}")
        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")
        progress_bar.progress((hour + 1) / 24)
    st.success(f"Fetched {len(all_balloon_data)} total balloon position entries.")
    return all_balloon_data

def get_weather_data(lat, lon, api_key):
    """
    Fetches wind speed and direction from OpenWeatherMap API for a given location.

    Args:
        lat (float): Latitude of the location.
        lon (float) : Longitude of the location.
        api_key (str): OpenWeatherMap API key.

    Returns:
        tuple: A tuple containing (wind_speed, wind_direction) in m/s and degrees,
               or (None, None) if the API call is unsuccessful or data is missing.
    """
    if not api_key:
        st.error("OpenWeatherMap API key not provided.")
        return None, None

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}lat={lat}&lon={lon}&appid={api_key}&units=metric"

    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "wind" in data:
            wind_speed = data["wind"].get("speed")
            wind_direction = data["wind"].get("deg")
            return wind_speed, wind_direction
        else:
            # st.warning(f"Wind data not found for lat={lat}, lon={lon}") # Suppress verbose logging
            return None, None

    except requests.exceptions.HTTPError as e:
        # st.warning(f"HTTP error fetching weather data for lat={lat}, lon={lon}: {e}") # Suppress verbose logging
        pass
    except requests.exceptions.ConnectionError as e:
        # st.warning(f"Connection error fetching weather data for lat={lat}, lon={lon}: {e}") # Suppress verbose logging
        pass
    except requests.exceptions.Timeout as e:
        # st.warning(f"Timeout error fetching weather data for lat={lat}, lon={lon}: {e}") # Suppress verbose logging
        pass
    except requests.exceptions.RequestException as e:
        # st.warning(f"Error fetching weather data for lat={lat}, lon={lon}: {e}") # Suppress verbose logging
        pass
    except json.JSONDecodeError:
        # st.warning(f"Malformed JSON response for lat={lat}, lon={lon}") # Suppress verbose logging
        pass
    except Exception as e:
        # st.warning(f"An unexpected error occurred fetching weather data for lat={lat}, lon={lon}: {e}") # Suppress verbose logging
        pass

    return None, None


def fetch_weather_for_all_balloon_data(all_balloon_data, api_key, limit=None):
    """Fetches weather data for each balloon entry, with an optional limit."""
    combined_data = []
    weather_fetch_count = 0
    successful_fetches = 0
    API_CALL_DELAY = 0.1

    st.info("Fetching weather data for balloon positions...")
    progress_bar = st.progress(0)

    data_to_process = all_balloon_data[:limit] if limit is not None else all_balloon_data

    if data_to_process:
        for i, entry in enumerate(data_to_process):
            lat = entry.get('lat')
            lon = entry.get('lon')

            try:
                lat = float(lat) if lat is not None else None
                lon = float(lon) if lon is not None else None
            except (ValueError, TypeError):
                lat = None
                lon = None

            if lat is not None and lon is not None:
                wind_speed, wind_direction = get_weather_data(lat, lon, api_key)
                weather_fetch_count += 1

                if wind_speed is not None and wind_direction is not None:
                    entry['wind_speed'] = wind_speed
                    entry['wind_direction'] = wind_direction
                    successful_fetches += 1
                else:
                    entry['wind_speed'] = None
                    entry['wind_direction'] = None

                combined_data.append(entry)
                time.sleep(API_CALL_DELAY)
            else:
                combined_data.append(entry) # Include entry even if lat/lon is missing

            progress_bar.progress((i + 1) / len(data_to_process))

        st.success(f"Attempted to fetch weather data for {weather_fetch_count} entries.")
        st.success(f"Successfully fetched weather data for {successful_fetches} entries.")
        st.success(f"Combined data contains {len(combined_data)} entries.")
    else:
        st.warning("No balloon data available for weather fetching.")

    return combined_data


def prepare_data_for_ml(combined_data):
    """Prepares combined data into a DataFrame for ML."""
    if not combined_data:
        st.warning("No combined data available to prepare for ML.")
        return pd.DataFrame()

    df = pd.DataFrame(combined_data)

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce')

    df.dropna(subset=['lat', 'lon', 'wind_speed', 'wind_direction'], inplace=True)

    if 'timestamp' in df.columns:
        df['timestamp_hour'] = df['timestamp'].str.extract(r'(\d+)H_ago').astype(float)
        df = df.sort_values(by=['id', 'timestamp_hour']) # Keep sorting for potential future use of diff features

        # Calculate movement features (not used for the current ML problem but kept for completeness)
        df['delta_lat'] = df.groupby('id')['lat'].diff()
        df['delta_lon'] = df.groupby('id')['lon'].diff()
        df['time_diff_hours'] = df.groupby('id')['timestamp_hour'].diff()
        df['displacement'] = (df['delta_lat']**2 + df['delta_lon']**2)**0.5
        df['speed_approx'] = df['displacement'] / df['time_diff_hours']
    else:
        df['timestamp_hour'] = None # Ensure timestamp_hour column exists

    df.dropna(subset=['lat', 'lon', 'timestamp_hour', 'wind_speed', 'wind_direction'], inplace=True) # Drop rows with NaNs in ML features/targets


    st.success(f"Prepared DataFrame for ML with {len(df)} entries after cleaning.")
    return df

def train_ml_model(df):
    """Trains the Random Forest Regressor model."""
    if df.empty:
        st.warning("No data available to train the ML model.")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = df[['lat', 'lon', 'timestamp_hour']]
    y = df[['wind_speed', 'wind_direction']]

    if X.empty or y.empty:
         st.warning("Features or target variables are empty after selection.")
         return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.info("Training Random Forest Regressor model...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    st.success("Model training complete.")

    return model, X_test, y_test, X_train # Return X_train for feature names


# --- Streamlit App ---

st.title("Windborne Balloon and Weather Data Analysis")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Acquisition & Preparation", "ML Model", "Visualizations", "Predict Wind"])

# --- Store state using Streamlit's session_state ---
if 'df_combined' not in st.session_state:
    st.session_state.df_combined = pd.DataFrame()
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = pd.DataFrame()
if 'y_test' not in st.session_state:
    st.session_state.y_test = pd.DataFrame()
if 'X_train' not in st.session_state:
     st.session_state.X_train = pd.DataFrame()


if page == "Home":
    st.header("Welcome!")
    st.write("""
    This application analyzes Windborne balloon data combined with OpenWeatherMap weather data.
    It performs data acquisition, cleaning, prepares data for Machine Learning, trains a model
    to predict wind speed and direction, and provides visualizations and a prediction interface.
    """)
    st.write("Use the sidebar to navigate through the different sections.")

elif page == "Data Acquisition & Preparation":
    st.header("Data Acquisition and Preparation")

    st.write("This section fetches balloon data from the Windborne API and weather data from OpenWeatherMap, then combines and prepares it for analysis and ML.")

    # Get OpenWeatherMap API key
    st.subheader("OpenWeatherMap API Key")
    openweathermap_api_key = st.text_input("Enter your OpenWeatherMap API Key", type="password")
    st.info("You can get an API key from the OpenWeatherMap website. Add it to your Colab secrets with the name `OPENWEATHERMAP_API_KEY` if running in Colab, or use environment variables in other environments.")

    # Add an input for the number of entries to process
    num_entries_to_process = st.number_input("Number of balloon entries to process for weather data (set to 0 for all)", min_value=0, value=100, step=10)


    if st.button("Run Data Acquisition and Preparation"):
        if not openweathermap_api_key:
            st.error("Please provide your OpenWeatherMap API key.")
        else:
            all_balloon_data = fetch_and_extract_all()
            if all_balloon_data:
                # Pass the limit to the weather fetching function
                limit = num_entries_to_process if num_entries_to_process > 0 else None
                combined_data_with_weather = fetch_weather_for_all_balloon_data(all_balloon_data, openweathermap_api_key, limit=limit)
                if combined_data_with_weather:
                    st.session_state.df_combined = prepare_data_for_ml(combined_data_with_weather)
                    if not st.session_state.df_combined.empty:
                        st.subheader("Prepared Data Preview")
                        st.dataframe(st.session_state.df_combined.head())
                else:
                    st.warning("No combined data with weather was generated.")
            else:
                st.warning("No balloon data was acquired.")

elif page == "ML Model":
    st.header("Machine Learning Model Training and Evaluation")

    if st.button("Train ML Model"):
        if not st.session_state.df_combined.empty:
            model, X_test, y_test, X_train = train_ml_model(st.session_state.df_combined)
            if model:
                st.session_state.ml_model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train # Store X_train for feature names
            else:
                st.warning("Model training failed.")
        else:
            st.warning("Please run the data acquisition and preparation step first.")

    if st.session_state.ml_model:
        st.subheader("Model Evaluation Results on Test Data:")
        if not st.session_state.X_test.empty and not st.session_state.y_test.empty:
            try:
                y_pred = st.session_state.ml_model.predict(st.session_state.X_test)

                y_pred_speed = y_pred[:, 0]
                y_pred_direction = y_pred[:, 1]

                y_test_speed = st.session_state.y_test['wind_speed']
                y_test_direction = st.session_state.y_test['wind_direction']

                mse_speed = mean_squared_error(y_test_speed, y_pred_speed)
                mse_direction = mean_squared_error(y_test_direction, y_pred_direction)
                r2_speed = r2_score(y_test_speed, y_pred_speed)
                r2_direction = r2_score(y_test_direction, y_pred_direction)

                st.write(f"**Wind Speed:**")
                st.write(f"  - Mean Squared Error (MSE): {mse_speed:.4f}")
                st.write(f"  - R-squared: {r2_speed:.4f}")
                st.write(f"**Wind Direction:**")
                st.write(f"  - Mean Squared Error (MSE): {mse_direction:.4f}")
                st.write(f"  - R-squared: {r2_direction:.4f}")

                if hasattr(st.session_state.ml_model, 'feature_importances_') and not st.session_state.X_train.empty:
                    st.subheader("Feature Importances")
                    feature_names = st.session_state.X_train.columns
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': st.session_state.ml_model.feature_importances_
                    })
                    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                    st.dataframe(feature_importance_df)
                else:
                     st.info("Feature importances not available.")

            except Exception as e:
                st.error(f"An error occurred during model evaluation: {e}")

        else:
            st.info("Test data not available for evaluation. Please train the model first.")
    else:
        st.info("ML model not trained yet.")


elif page == "Visualizations":
    st.header("Visualizations")

    st.subheader("Combined Data Visualization (Actual Wind)")
    if not st.session_state.df_combined.empty:
        st.subheader("Filter Data")
        with st.form("filter_form_viz"):
            col1, col2 = st.columns(2)
            with col1:
                min_lat_viz = st.text_input("Min Latitude", key="min_lat_viz")
                max_lat_viz = st.text_input("Max Latitude", key="max_lat_viz")
            with col2:
                min_lon_viz = st.text_input("Min Longitude", key="min_lon_viz")
                max_lon_viz = st.text_input("Max Longitude", key="max_lon_viz")
            hour_filter_viz = st.text_input("Hour (0-23)", key="hour_filter_viz")

            # Add the submit button to the form
            apply_filter_viz = st.form_submit_button("Apply Filter")
            clear_filter_viz = st.form_submit_button("Clear Filter")


        filtered_df_viz = st.session_state.df_combined.copy()

        if apply_filter_viz:
            try:
                if min_lat_viz:
                    filtered_df_viz = filtered_df_viz[filtered_df_viz['lat'] >= float(min_lat_viz)]
                if max_lat_viz:
                    filtered_df_viz = filtered_df_viz[filtered_df_viz['lat'] <= float(max_lat_viz)]
                if min_lon_viz:
                    filtered_df_viz = filtered_df_viz[filtered_df_viz['lon'] >= float(min_lon_viz)]
                if max_lon_viz:
                    filtered_df_viz = filtered_df_viz[filtered_df_viz['lon'] <= float(max_lon_viz)]
                if hour_filter_viz:
                     hour_float_viz = float(hour_filter_viz)
                     if 'timestamp_hour' in filtered_df_viz.columns and not filtered_df_viz['timestamp_hour'].isnull().all():
                          filtered_df_viz = filtered_df_viz[filtered_df_viz['timestamp_hour'] == hour_float_viz]
                     else:
                          filtered_df_viz = filtered_df_viz.head(0)

            except ValueError:
                st.error("Invalid filter input. Please provide numeric values.")
                filtered_df_viz = st.session_state.df_combined.head(0)
            except Exception as e:
                st.error(f"An error occurred during filtering: {e}")
                filtered_df_viz = st.session_state.df_combined.head(0)

        st.subheader(f"Balloon Positions ({len(filtered_df_viz)} entries) with Actual Wind Speed (Filtered)")
        if not filtered_df_viz.empty and 'lat' in filtered_df_viz.columns and 'lon' in filtered_df_viz.columns and 'wind_speed' in filtered_df_viz.columns:
            fig_base_data = px.scatter_mapbox(filtered_df_viz,
                                    lat="lat",
                                    lon="lon",
                                    color="wind_speed",
                                    size="wind_speed",
                                    hover_name="id",
                                    hover_data={"lat": True, "lon": True, "wind_speed": True, "wind_direction": True, "timestamp": True},
                                    color_continuous_scale=px.colors.cyclical.IceFire,
                                    size_max=15,
                                    zoom=1,
                                    height=500)

            fig_base_data.update_layout(mapbox_style="open-street-map")
            fig_base_data.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_base_data, use_container_width=True)
        else:
             st.info("No combined data available or matching filter criteria for base visualization.")

    else:
        st.info("No combined data available. Please run the data acquisition and preparation step.")


    st.subheader("Machine Learning Results Visualizations (on Test Data)")
    if st.session_state.ml_model and not st.session_state.X_test.empty and not st.session_state.y_test.empty:
        try:
            y_pred_viz = st.session_state.ml_model.predict(st.session_state.X_test)
            y_pred_df_viz = pd.DataFrame(y_pred_viz, index=st.session_state.X_test.index, columns=['predicted_wind_speed', 'predicted_wind_direction'])
            visualization_df_ml = pd.concat([st.session_state.X_test, st.session_state.y_test, y_pred_df_viz], axis=1)
            visualization_df_ml.rename(columns={
                'wind_speed': 'actual_wind_speed',
                'wind_direction': 'actual_wind_direction'
            }, inplace=True)

            visualization_df_ml['speed_difference'] = visualization_df_ml['predicted_wind_speed'] - visualization_df_ml['actual_wind_speed']
            visualization_df_ml['direction_difference'] = visualization_df_ml['predicted_wind_direction'] - visualization_df_ml['actual_wind_direction']

            st.subheader("Wind Speed Prediction Difference (Predicted - Actual)")
            fig_speed_diff = px.scatter_mapbox(visualization_df_ml,
                                               lat="lat",
                                               lon="lon",
                                               color="speed_difference",
                                               size="actual_wind_speed",
                                               hover_name=visualization_df_ml.index,
                                               hover_data={"lat": True, "lon": True, "timestamp_hour": True, "actual_wind_speed": ':.2f', "predicted_wind_speed": ':.2f', "speed_difference": ':.2f', "actual_wind_direction": True, "predicted_wind_direction": True, "direction_difference": ':.2f'},
                                               color_continuous_scale="RdBu",
                                               size_max=10,
                                               zoom=1,
                                               height=600)
            fig_speed_diff.update_layout(mapbox_style="open-street-map")
            fig_speed_diff.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_speed_diff, use_container_width=True)

            st.subheader("Wind Direction Prediction Difference (Predicted - Actual)")
            fig_direction_diff = px.scatter_mapbox(visualization_df_ml,
                                               lat="lat",
                                               lon="lon",
                                               color="direction_difference",
                                               size="actual_wind_direction",
                                               hover_name=visualization_df_ml.index,
                                               hover_data={"lat": True, "lon": True, "timestamp_hour": True, "actual_wind_speed": ':.2f', "predicted_wind_speed": ':.2f', "speed_difference": ':.2f', "actual_wind_direction": True, "predicted_wind_direction": True, "direction_difference": ':.2f'},
                                               color_continuous_scale="RdBu",
                                               size_max=10,
                                               zoom=1,
                                               height=600)
            fig_direction_diff.update_layout(mapbox_style="open-street-map")
            fig_direction_diff.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_direction_diff, use_container_width=True)

            st.subheader("Actual vs. Predicted Wind Speed")
            fig_speed_scatter = px.scatter(visualization_df_ml,
                                           x="actual_wind_speed",
                                           y="predicted_wind_speed",
                                           hover_data={"lat": True, "lon": True, "timestamp_hour": True, "speed_difference": ':.2f'},
                                           title="Actual vs. Predicted Wind Speed on Test Data")
            st.plotly_chart(fig_speed_scatter, use_container_width=True)

            st.subheader("Actual vs. Predicted Wind Direction")
            fig_direction_scatter = px.scatter(visualization_df_ml,
                                              x="actual_wind_direction",
                                              y="predicted_wind_direction",
                                              hover_data={"lat": True, "lon": True, "timestamp_hour": True, "direction_difference": ':.2f'},
                                              title="Actual vs. Predicted Wind Direction on Test Data")
            st.plotly_chart(fig_direction_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating ML results visualizations: {e}")

    else:
        st.info("ML test data or predictions not available for visualization. Please train the model first.")


elif page == "Predict Wind":
    st.header("Get Wind Prediction")
    st.write("Enter location and hour to get wind speed and direction prediction from the trained model.")

    if st.session_state.ml_model:
        lat_input = st.text_input("Latitude", value="0.0", key="predict_lat")
        lon_input = st.text_input("Longitude", value="0.0", key="predict_lon")
        hour_input = st.text_input("Hour (0-23)", value="0", key="predict_hour")

        if st.button("Predict Wind", key="predict_button"):
            try:
                lat = float(lat_input)
                lon = float(lon_input)
                hour = float(hour_input)

                if not (0 <= hour <= 23):
                    st.error("Invalid hour value. Please provide a number between 0 and 23.")
                else:
                    input_data = pd.DataFrame([[lat, lon, hour]], columns=['lat', 'lon', 'timestamp_hour'])
                    prediction = st.session_state.ml_model.predict(input_data)
                    predicted_speed = prediction[0, 0]
                    predicted_direction = prediction[0, 1]

                    st.subheader("Prediction Result:")
                    st.write(f"Predicted Wind Speed: {predicted_speed:.2f} m/s")
                    st.write(f"Predicted Wind Direction: {predicted_direction:.2f} degrees")

            except ValueError:
                st.error("Invalid input. Please provide numeric values for latitude, longitude, and hour.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("ML model not trained yet. Please go to the 'ML Model' page to train the model.")
