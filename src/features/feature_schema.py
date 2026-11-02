"""Names and definitions shared by Spark and the LSTM inference process."""
HISTORY_FEATURES = [
    "demand_scaled", "temperature_scaled",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos",
]
FUTURE_FEATURES = [
    "future_temperature_scaled",
    "temperature_rolling_mean_1h_scaled",
    "temperature_rolling_mean_24h_scaled",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos",
]
