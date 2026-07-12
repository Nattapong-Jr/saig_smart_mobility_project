import pandas as pd

df = pd.read_csv("cleaned_data.csv")

df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.day_name()
df["month"] = df["time"].dt.month

print(df[["time", "hour", "day_of_week", "month"]].head(10))

def get_time_period(hour):
    if 5 <= hour < 12:
        return "เช้า"
    elif 12 <= hour < 17:
        return "บ่าย"
    elif 17 <= hour < 21:
        return "เย็น"
    else:
        return "กลางคืน"

df["time_period"] = df["hour"].apply(get_time_period)
print("\n--- จำนวนอุบัติเหตุแต่ละชาวงเวลา ---")
print(df["time_period"].value_counts()) 