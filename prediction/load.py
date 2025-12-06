def predict_load(user_input):
    import pandas as pd
    import pickle
    import os


    BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "models-encoders")
    BASE_PATH = os.path.abspath(BASE_PATH)


    with open(os.path.join(BASE_PATH, "load-encoder.pickle"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_PATH, "load-ohe.pickle"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(BASE_PATH, "load-model.pickle"), "rb") as f:
        final_model = pickle.load(f)

    EXPECTED_COLS = [
        "temp_C", "humidity_%", "wind_speed_m_s", "solar_irradiance_W_m2",
        "precip_mm", "year", "month", "day", "hour",
        "weather_Cloudy", "weather_Rainy", "weather_Sunny"
    ]

    df = pd.DataFrame([user_input])

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour

    df = df.drop(columns=["datetime"], errors="ignore")

    weather_ohe = encoder.transform(df[["weather"]])
    weather_df = pd.DataFrame(
        weather_ohe,
        columns=encoder.get_feature_names_out(["weather"])
    )

    df = pd.concat([df.drop(columns=["weather"]), weather_df], axis=1)

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[EXPECTED_COLS]

    X_scaled = scaler.transform(df)
    prediction = final_model.predict(X_scaled)

    return prediction[0]
