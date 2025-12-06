def predict_wind(user_input):
    import pickle
    import pandas as pd
    import os

    BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "models-encoders")
    BASE_PATH = os.path.abspath(BASE_PATH)

    # --------------------------------------------
    # Load saved objects
    # --------------------------------------------
    with open(os.path.join(BASE_PATH, "wind-encoder.pickle"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_PATH, "wind-ohe.pickle"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(BASE_PATH, "wind-model.pickle"), "rb") as f:
        wind_model = pickle.load(f)

    # Columns used in training
    EXPECTED_COLS = [
        "temp_C", "humidity_%", "wind_speed_m_s",
        "precip_mm", "installed_wind_MW",
        "year", "month", "day", "hour",
        "weather_Cloudy", "weather_Rainy", "weather_Sunny"
    ]

    # --------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------
    df = pd.DataFrame([user_input])

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df = df.drop(columns=["datetime"], errors="ignore")

    # --------------------------------------------
    # One-hot encode weather
    # --------------------------------------------
    weather_encoded = encoder.transform(df[["weather"]])
    weather_df = pd.DataFrame(
        weather_encoded,
        columns=encoder.get_feature_names_out(["weather"])
    )

    df = pd.concat([df.drop(columns=["weather"]), weather_df], axis=1)

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
    df_scaled = scaler.transform(df)

    prediction = wind_model.predict(df_scaled)

    return prediction[0]
