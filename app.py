import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(page_title="Smart Mobility - จุดเสี่ยงอุบัติเหตุ", layout="wide")

st.title("Smart Mobility: แผนที่จุดเสี่ยงอุบัติเหตุ")
st.write("วิเคราะห์จากข้อมูลอุบัติเหตุจริงทั่วประเทศไทย")

df = pd.read_csv("cleaned_data.csv")

st.write(f"ข้อมูลทั้งหมด: {len(df):,} เหตุการณ์")
st.dataframe(df.head())

st.header("แปนที่จุดเสี่ยงอุบัติเหตุ (Heatmap)")

sample_size = st.slider("จำนวนจุดที่แสดงบนแผนที่", 1000, 2000, 5000, step=1000)
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

m = folium.Map(location=[13.7563, 100.5018], zoom_start=6, tiles="CartoDB dark_matter")

heat_data = df_sample[["latitude", "longitude"]].values.tolist()
HeatMap(heat_data, radius=8, blur=10).add_to(m)

st_folium(m, width=1200, height=600)
