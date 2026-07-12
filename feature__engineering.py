import pandas as pd

df = pd.read_csv("cleaned_data.csv")

df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.day_name()
df["month"] = df["time"].dt.month

print(df[["time", "hour", "day_of_week", "month"]].head(10))