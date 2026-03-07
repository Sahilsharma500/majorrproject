import streamlit as st
from datetime import datetime



from prediction.load import predict_load
from prediction.solar import predict_solar
from prediction.wind import predict_wind


# --------------------------
# STREAMLIT UI
# --------------------------
st.title("⚡ Energy Generation & Load Forecasting Dashboard")

st.write("Provide the input parameters below for Load, Solar, and Wind forecasting.")

# --- USER INPUTS ---
st.subheader("📥 Input Parameters")

user_datetime = st.text_input(
    "Date & Time (YYYY-MM-DD HH:MM:SS)", 
    value="2025-10-13 10:00:00"
)

weather = st.selectbox(
    "Weather Condition",
    ["Sunny", "Cloudy", "Rainy"]
)

temp_C = st.number_input("Temperature (°C)", value=30.0)
humidity = st.number_input("Humidity (%)", value=55.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=4.2)
solar_irradiance = st.number_input("Solar Irradiance (W/m²)", value=400.0)
precip_mm = st.number_input("Precipitation (mm)", value=0.0)

# Solar model specific
installed_solar_MW = st.number_input("Installed Solar Capacity (MW)", value=3000.0)
panel_area_m2 = st.number_input("Total Solar Panel Area (m²)", value=12750000.0)
num_panels = st.number_input("Number of Solar Panels", value=7500000)

# Wind model specific
installed_wind_MW = st.number_input("Installed Wind Capacity (MW)", value=10.0)


# --- CONVERT TO DICTIONARIES FOR EACH MODEL ---
load_input = {
    "datetime": user_datetime,
    "weather": weather,
    "temp_C": temp_C,
    "humidity_%": humidity,
    "wind_speed_m_s": wind_speed,
    "solar_irradiance_W_m2": solar_irradiance,
    "precip_mm": precip_mm,
    "installed_solar_MW": installed_solar_MW,
    "panel_area_m2": panel_area_m2,
    "num_panels": num_panels,
    "installed_wind_MW": installed_wind_MW
}

solar_input = {
    "datetime": user_datetime,
    "weather": weather,
    "temp_C": temp_C,
    "humidity_%": humidity,
    "wind_speed_m_s": wind_speed,
    "solar_irradiance_W_m2": solar_irradiance,
    "precip_mm": precip_mm,
    "installed_solar_MW": installed_solar_MW,
    "panel_area_m2": panel_area_m2,
    "num_panels": num_panels
}

wind_input = {
    "datetime": user_datetime,
    "weather": weather,
    "temp_C": temp_C,
    "humidity_%": humidity,
    "wind_speed_m_s": wind_speed,
    "precip_mm": precip_mm,
    "installed_wind_MW": installed_wind_MW
}


# --- PREDICT ---
if st.button("🔮 Predict"):
    load_pred = predict_load(load_input)
    solar_pred = predict_solar(solar_input)
    wind_pred = predict_wind(wind_input)

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⚡ Load Demand (MW)", f"{load_pred:.2f}")

    with col2:
        st.metric("☀ Solar Generation (MW)", f"{solar_pred:.2f}")

    with col3:
        st.metric("🌬 Wind Generation (MW)", f"{wind_pred:.2f}")

    st.success("✔ Predictions computed successfully!")
