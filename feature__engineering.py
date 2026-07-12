import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("cleaned_data.csv")

df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.day_name()
df["month"] = df["time"].dt.month

print(df[["time", "hour", "day_of_week", "month"]].head(10))

#check accident defferent time
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

df["severity_score"] = df["fatalities"] * 3 + df["injuries"] * 1

def get_severity_level(score):
    if score == 0:
        return "ต่ำ"
    elif score <= 2:
        return "กลาง"
    else:
        return "สูง"

df["severity_level"] = df["severity_score"].apply(get_severity_level)
print("\n--- จำนวนอุบัติเหตุแต่ละความรุนแรง ----")
print(df["severity_level"].value_counts())

feature_columns = ["hour", "month", "time_period", "day_of_week", "weather", "cause_grouped",
                   "road_shape", "terrain", "first_vehicle"]

df_model = df[feature_columns + ["severity_level"]].copy()

label_encoders = {}
for col in ["time_period", "day_of_week", "weather", "cause_grouped", "road_shape", "terrain", "first_vehicle"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

print(df_model.head())
print("\nขนาดข้อมูลพร้มอใช้:", df_model.shape)

X = df_model[feature_columns]
y = df_model["severity_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_perd = model.predict(X_test)

print("\n--- ผมทดสอบโมเดล ----")
print("Accuracy: ", accuracy_score(y_test, y_perd))
print("\nรายละเอียดเพิ่มเติม: ")
print(classification_report(y_test, y_perd))