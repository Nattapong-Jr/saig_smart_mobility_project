import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Smart Mobility - จุดเสี่ยงอุบัติเหตุ", layout="wide")

st.title("Smart Mobility: แผนที่จุดเสี่ยงอุบัติเหตุ")
st.write("วิเคราะห์จากข้อมูลอุบัติเหตุจริงทั่วประเทศไทย")

df = pd.read_csv("cleaned_data.csv")

st.write(f"ข้อมูลทั้งหมด: {len(df):,} เหตุการณ์")
st.dataframe(df.head())