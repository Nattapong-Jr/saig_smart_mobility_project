import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np

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

importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n--- Feature Importance ---")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("Feature Importance - Accident Severity Model")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\nบันทึกกราฟ feature_importance.png")

result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)

print("\n--- permutation importance (แม่นยำกว่า)---")
print(perm_df)

feature_columns_v2 = ["first_vehicle", "cause_grouped", "month", "day_of_week",
                      "weather", "road_shape", "terrain"]
X_v2 = df_model[feature_columns_v2]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_v2, y, test_size=0.2, random_state=42)

model_v2 = RandomForestClassifier(n_estimators=100, random_state=42)
model_v2.fit(X_train2, y_train2)

y_perd2 = model_v2.predict(X_test2)

print("\n--- ผลการทดสอบโฒเดล v2 (ตัด feature ที่ไม่มีผล) ---")
print("Accuracy:", accuracy_score(y_test2, y_perd2))
print(classification_report(y_test2, y_perd2))

day_order = {"Monday": 0, "Tuesday":1, "Wednesday":2, "Thursday": 3,
             "Friday": 4, "Saturday": 5, "Sunday": 6}
df_model["day_num"] = df["day_of_week"].map(day_order)

df_model["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df_model["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df_model["day_sin"] = np.sin(2 * np.pi * df_model["day_num"] / 7)
df_model["day_cos"] = np.cos(2 * np.pi * df_model["day_num"] / 7)

df_model["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df_model["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

print(df_model[["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]].head())

feature_columns_cyclic = ["first_vehicle", "cause_grouped", "weather", "road_shape", "terrain",
                          "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]

X_cyclic = df_model[feature_columns_cyclic]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_cyclic, y, test_size=0.2, random_state=42)

model_v4 = RandomForestClassifier(n_estimators=100, random_state=42)
model_v4.fit(X_train3, y_train3)

y_pred4 = model_v4.predict(X_test3)

print("\n--- โมเดล v4 (cyclic ending) ---")
print("Accuracy:", accuracy_score(y_test3, y_pred4))
print(classification_report(y_test3, y_pred4))

recall_high = recall_score(y_test3, y_pred4, labels=["สูง"], average="micro")
print("Recall เฉพาะกลุ่ม 'สูง':", recall_high)

model_v5 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model_v5.fit(X_train3, y_train3)

y_pred5 = model_v5.predict(X_test3)

print("\n--- โมเดล v5 (cyclic + class_weight=balanced) ---")
print("Accuracy:", accuracy_score(y_test3, y_pred5))
print(classification_report(y_test3, y_pred5))
