def predict_solar(user_input):
    import pickle
    import pandas as pd
    import os

    BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "models-encoders")
    BASE_PATH = os.path.abspath(BASE_PATH)

    # --------------------------------------------
    # Load saved objects
    # --------------------------------------------
    with open(os.path.join(BASE_PATH, "solar-encoder.pickle"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_PATH, "solar-ohe.pickle"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(BASE_PATH, "solar-model.pickle"), "rb") as f:
        final_model = pickle.load(f)

    # Expected column order during training
    EXPECTED_COLS = [
        'temp_C', 'humidity_%', 'wind_speed_m_s', 'solar_irradiance_W_m2',
        'installed_solar_MW', 'panel_area_m2', 'num_panels', 'year',
        'weather_Cloudy', 'weather_Rainy', 'weather_Sunny'
    ]

    # --------------------------------------------
    # Input Processing
    # --------------------------------------------
    df = pd.DataFrame([user_input])

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    df = df.drop(columns=['datetime', 'month', 'day', 'hour', 'precip_mm'], errors='ignore')

    # --------------------------------------------
    # One-hot encode weather
    # --------------------------------------------
    encoded_weather = encoder.transform(df[['weather']])
    weather_df = pd.DataFrame(
        encoded_weather,
        columns=encoder.get_feature_names_out(['weather'])
    )

    df = pd.concat([df.drop(columns=['weather']), weather_df], axis=1)

    # --------------------------------------------
    # Add missing columns
    # --------------------------------------------
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[EXPECTED_COLS]

    # --------------------------------------------
    # Scale & Predict
    # --------------------------------------------
    X_scaled = scaler.transform(df)
    prediction = final_model.predict(X_scaled)

    return prediction[0]
