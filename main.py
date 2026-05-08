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
wind_speed = st.number_input("Wind Speed (m/s)", value=5.7)
solar_irradiance = st.number_input("Solar Irradiance (W/m²)", value=400.0)
precip_mm = st.number_input("Precipitation (mm)", value=0.0)

# Solar model specific
installed_solar_MW = st.number_input("Installed Solar Capacity (MW)", value=3000.0)
panel_area_m2 = st.number_input("Total Solar Panel Area (m²)", value=12750000.0)
num_panels = st.number_input("Number of Solar Panels", value=7500000)

# Wind model specific
installed_wind_MW = st.number_input("Installed Wind Capacity (MW)", value=50.0)


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
    
    # Store predictions in session state so they persist when sliders change
    st.session_state['load_pred'] = load_pred
    st.session_state['solar_pred'] = solar_pred
    st.session_state['wind_pred'] = wind_pred

# --------------------------
# GRID OPTIMIZATION MODULE
# --------------------------
st.markdown("---")
st.subheader("🔋 Smart Grid Optimizer (Time-of-Use Arbitrage)")
st.write("Tune your Battery specifications below and run the advanced monetary optimizer.")

col_b1, col_b2 = st.columns(2)
with col_b1:
    slider_cap = st.slider("Battery Capacity (MWh)", min_value=100, max_value=5000, value=1000, step=100)
with col_b2:
    slider_mw = st.slider("Max Charge/Discharge Rate (MW)", min_value=10, max_value=1000, value=300, step=10)

if st.button("🚀 Run Grid Optimizer"):
    if 'load_pred' not in st.session_state:
        st.warning("Please click 'Predict' to compute base ML generation values first!")
    else:
        import math
        import numpy as np
        import plotly.express as px
        from grid.cost_optimizer import run_grid_optimization
        
        load_pred = st.session_state['load_pred']
        solar_pred = st.session_state['solar_pred']
        wind_pred = st.session_state['wind_pred']
        
        # 1. Synthesize 24-hour profiles
        hours = np.arange(24)
        
        # Solar: Sine wave from 6 AM to 6 PM (hour 18)
        solar_profile = np.zeros(24)
        for h in range(6, 19):
            solar_profile[h] = solar_pred * math.sin((h - 6) * math.pi / 12)
            
        # Wind: Random +/- 10% noise around prediction
        np.random.seed(42)  # For consistent graph reloading
        wind_profile = wind_pred * np.random.uniform(0.9, 1.1, size=24)
        
        # Load: Empirical residential/commercial load curve
        load_shape = np.array([
            0.5, 0.4, 0.4, 0.4, 0.5, 0.6,   # Night 0-5
            0.8, 1.0, 0.9, 0.8, 0.9, 1.0,   # Morning 6-11
            1.1, 1.0, 0.9, 0.9, 1.0, 1.2,   # Noon 12-17
            1.5, 1.6, 1.4, 1.2, 0.9, 0.7    # Evening 18-23
        ])
        # Scale load profile so the peak matches the user's ML prediction
        load_profile = load_pred * (load_shape / np.max(load_shape))
        
        with st.spinner("Running CVXPY Linear Programming Optimizer..."):
            res = run_grid_optimization(load_profile, solar_profile, wind_profile, slider_cap, slider_mw)
        
        if res is None:
            st.error("Optimization failed! Mathematical parameters are infeasible.")
        else:
            df = res['df']
            
            def format_inr(value):
                abs_val = abs(value)
                if abs_val >= 1e7:
                    return f"{value/1e7:,.2f} Cr"
                elif abs_val >= 1e5:
                    return f"{value/1e5:,.2f} L"
                else:
                    return f"{value:,.2f}"

            # Metric Cards
            st.markdown("### 💰 Economic Results")
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.metric("Cost (Grid-Only)", f"₹ {format_inr(res['baseline_cost'])}")
            with scol2:
                if res['total_cost'] < 0:
                    st.metric("Cost (Grid + Battery)", f"₹ {format_inr(-res['total_cost'])}")
                else:
                    st.metric("Cost (Grid + Battery)", f"₹ {format_inr(res['total_cost'])}")
            with scol3:
                # Savings can be very large, parse them nicely too
                st.metric("Total Savings", f"₹ {format_inr(res['savings'])}", delta=f"{res['savings']:,.0f}", delta_color="normal")
            with scol4:
                st.metric("Total Grid Import", f"{res['total_import']:,.2f} MWh")
                
            # Line Charts
            st.markdown("### 📈 Power Grid Dispatch Timeline")
            # Melt dataframe for easy multi-line plotting in streamlit
            chart_df = df[['Hour', 'Load (MW)', 'Solar (MW)', 'Wind (MW)', 'Grid Import (MW)']].set_index('Hour')
            fig1 = px.line(chart_df, title="Power Grid Dispatch Timeline")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("### 🔋 Battery Autonomy (State of Charge %)")
            fig2 = px.line(df, x='Hour', y='SOC (%)', title="Battery Autonomy (%)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # DataFrame
            st.markdown("### 📋 Hourly Dispatch Data")
            st.dataframe(df.style.highlight_max(axis=0))
