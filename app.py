import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(page_title="Smart Mobility - จุดเสี่ยงอุบัติเหตุ", layout="wide")

st.title("Smart Mobility: แผนที่จุดเสี่ยงอุบัติเหตุ")
st.write("วิเคราะห์จากข้อมูลอุบัติเหตุจริงทั่วประเทศไทย")

df = pd.read_csv("data_with_feature.csv")

st.write(f"ข้อมูลทั้งหมด: {len(df):,} เหตุการณ์")
st.dataframe(df.head())

st.header("แผนที่จุดเสี่ยงอุบัติเหตุ (Heatmap)")

st.header("ข้อสรุปสำคัญ")

col1, col2, col3 = st.columns(3)

with col1:
    top_cause = df["cause_grouped"].value_counts().idxmax()
    top_cause_pct = df["cause_grouped"].value_counts(normalize=True).max() * 100
    st.metric("สาเหตุอันดับ 1", top_cause, f"{top_cause_pct:.1f}% ของทั้งหมด")

with col2:
    top_shape = df["road_shape"].value_counts().idxmax()
    st.metric("ลักษณะถนนที่เกิดเหตุบ่อยสุด", top_shape)

with col3:
    severe_count = (df["severity_level"] == "สูง").sum()
    severe_pct = severe_count / len(df) * 100
    st.metric("อุบัติเหตุรุนแรง", f"{severe_count} ครั้ง", f"{severe_pct:.1f}% ของทั้งหมด")

sample_size = st.slider("จำนวนจุดที่แสดงบนแผนที่", 1000, 2000, 5000, step=1000)
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

m = folium.Map(location=[13.7563, 100.5018], zoom_start=6, tiles="CartoDB dark_matter")

heat_data = df_sample[["latitude", "longitude"]].values.tolist()
HeatMap(heat_data, radius=8, blur=10).add_to(m)

st_folium(m, width=1200, height=600)

st.header("ทำนายระดับความเสี่ยงอุบัติเหตุ")

st.warning(
    "⚠️ **ข้อควรรู้:** โมเดลนี้มีความแม่นยำโดยรวมประมาณ 51% และถูกออกแบบให้ "
    "**เน้นจับกรณีอุบัติเหตุรุนแรง (recall สูง)** เพื่อความปลอดภัย ผลลัพธ์จึงอาจ "
    "'เตือนเกินจำเป็น' ได้บ่อยกว่าปกติ ควรใช้เป็นข้อมูลประกอบการตัดสินใจเท่านั้น "
    "ไม่ใช่คำทำนายที่แม่นยำสมบูรณ์"
)

col1, col2, = st.columns(2)

with col1:
    input_venicle = st.selectbox("ประเภทยานพาหนะ", df["first_vehicle"].unique())
    input_cause = st.selectbox("สาเหตุ (คาดการณ์)", df["cause_grouped"].unique())
    input_weather = st.selectbox("สภาพอากาศ", df["weather"].unique())

with col2:
    input_road_shape = st.selectbox("ลักษณะถนน", df["road_shape"].unique())
    input_terrain = st.selectbox("ภูมิประเทศ", df["terrain"].unique())

if st.button("ทำนายความเสี่ยง", type="primary"):
    st.write("กำลังพัฒนาส่วนทำนายในขั้นตอนถัดไป...")
