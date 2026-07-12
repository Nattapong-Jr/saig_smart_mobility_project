import pandas as pd

df = pd.read_csv("thai_accidental_dataset.csv")

print(df.head())
print("151779, 13:", df.shape)
print(df.isnull().sum())

print("\n--- ค่าที่เป็นไปได้ในแต่ละคอลัมน์ ---")
print("\nweather")
print(df["weather"].value_counts())

print("\ncause:")
print(df["cause"].value_counts())

print("\nroad_charactereistic:")
print(df["road_characteristic"].value_counts())

print("\n---  รายการ cause ทั้งหมด (ไม่ตัด) ---")
pd.set_option('display.max_rows', None)
print(df["cause"].value_counts())

def group_cause(text):
    if pd.isna(text):
        return "ไม่ระบุ"

    keyword_groups = {
        "ขับเร็วเกินกำหนด": ["เร็วเกิน"],
        "ตัดหน้า/กระชั้นชอด": ["ตัดหน้า", "กระชั้นชิด"],
        "หลับใน/เมื่อยล้า": ["หลับใน", "เมื่อยล้า"],
        "เมาสุรา/สารเสพติด": ["เมา", "แอลกอฮอล์", "สารออกฤทธิ์"],
        "ยานพาหนะขัดข้อง": ["ยาง", "เบรค", "เบรก", "เบคร", "ระบบ", "เครื่องยนต์", 
                            "พวงมาลัย", "ไฟฟ้า", "อุปกรณ์ยาพาหนะ", "หางพ่วง"],
        "ฝ่าฝืนกฎจราจร": ["ฝ่าฝืน", "ย้อนศร", "ผิดกฎหมาย", "ผิดช่องทาง", "ไม่ให้สัญาณ", "ไม่ให้สิทธิ"],
        "สภาพถนน/สิ่งแวดล้อม": ["ถนนลื่น", "ถนนชำรุด", "ถนนแคบ", "ทางโค้งอันตราย", "สิ่งกีดขวาง",
                                "แสงสว่าง", "ทัศนวิสัย", "มองเห็น", "คราบสะสม", "ผิวถนน"],
        "ไม่ชำนาญ/ไม่คุ้นทาง": ["ไม่ชำนาญ", "ไม่คุ้นเคย" , "ไม่คุ้นชิน"],
        "เสียสมธิ": ["โทรศัพท์", "สิ่งรบกวน", "ละสายตา"],
        "สุขภาพ/ร่างกาย": ["โรคประจำตัว", "ป่วย", "สายตา", "ไร้ความสามารถ"],
        "พฤติกรรมเสี่ยงอื่นๆ": ["บรรทุกเกิน", "สูยเสียการควบคุม", "รถหยุดกระทันหัน", "เปลี่ยนช่องทางกะทันหัน", "แซง"],
    }

    for group_name, keyword in keyword_groups.items():
        for kw in keyword:
            if kw in text:
                return group_name
    return "ไม่ระบุ/อื่นๆ"


df["cause_grouped"] = df["cause"].apply(group_cause)
print("\n--- ผลลัพธ์หลังจัดกลุ่ม cause ---")
print(df["cause_grouped"].value_counts())

def split_road(text):
    if pd.isna(text):
        return pd.Series(["ไม่ระบุ","ไม่ระบุ"])
    parts = text.split("+")
    if len(parts) == 2:
        return pd.Series([parts[0],parts[1]])
    else:
        return pd.Series([text, "ไม่ระบุ"])

df[["road_shape", "terrain"]] = df["road_characteristic"].apply(split_road)

print("\n--- road_shape (รูปแบบถนน) ---")
print(df["road_shape"].value_counts())

print("\n--- terrain ภูมิประเทศ ---")
print(df["terrain"].value_counts())

df["terrain"] = df["terrain"].replace({
    "ไม่มีความลาดชัน/ที่ราบ":"ไม่มีความลาดชัน", "": "ไม่ระบุ"
})

print("\n--- terrain clean ---")
print(df["terrain"].value_counts())

columns_to_fill = ["route", "province", "first_vehicle", "accident_type", "weather"]
for col in columns_to_fill:
    df[col] = df[col].fillna("ไม่ระบุ")

print("\n--- เช็ค missing values หลังเติมค่า ---")
print(df.isnull().sum())

df.to_csv("cleaned_data.csv", index=False)
print("\nsave cleaned_data.csv success")