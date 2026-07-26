import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import joblib
import numpy as np
from google import genai
import os
import json

st.set_page_config(page_title="Smart Mobility - จุดเสี่ยงอุบัติเหตุ", layout="wide")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_law(question, top_k=3):
    result = rag_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    question_embedding = np.array(result.embeddings[0].values)
    similarities = [cosine_similarity(question_embedding, emb) for emb in law_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [law_chunks[i] for i in top_indices]

def answer_law_question(question):
    relevant_chunks = search_law(question)
    context = "\n\n".join([f"[หน้า {c['page']}]\n{c['text']}" for c in relevant_chunks])
    
    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับกฎหมายจราจรไทย ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม
ถ้าข้อมูลไม่พอตอบ ให้บอกตรงๆ ว่าไม่พบข้อมูลที่เกี่ยวข้อง อย่าเดาเอง

ข้อมูลอ้างอิง:
{context}

คำถาม: {question}

ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย พร้อมระบุว่าอ้างอิงจากหน้าไหน"""

    response = rag_client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text, relevant_chunks

st.title("Smart Mobility: แผนที่จุดเสี่ยงอุบัติเหตุ")
st.write("วิเคราะห์จากข้อมูลอุบัติเหตุจริงทั่วประเทศไทย")

df = pd.read_csv("data_with_feature.csv")

model = joblib.load("accident_severity_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

#RAG
rag_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

with open("law_chunks.json", "r", encoding="utf-8") as f:
    law_chunks = json.load(f)

law_embeddings = np.load("law_embeddings.npy")

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
    input_vehicle = st.selectbox("ประเภทยานพาหนะ", df["first_vehicle"].unique())
    input_cause = st.selectbox("สาเหตุ (คาดการณ์)", df["cause_grouped"].unique())
    input_weather = st.selectbox("สภาพอากาศ", df["weather"].unique())

with col2:
    input_road_shape = st.selectbox("ลักษณะถนน", df["road_shape"].unique())
    input_terrain = st.selectbox("ภูมิประเทศ", df["terrain"].unique())
    input_hour = st.slider("ชั่วโมง (0-23 น.)", 0, 23, 12)
    input_day = st.selectbox(" วันในสัปดาห์", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                               "Friday", "Saturday", "Sunday",])
    input_month = st.selectbox("เดือน", list(range(1, 13)))

if st.button("ทำนายความเสี่ยง", type="primary"):
    encoded_vehicle = label_encoders["first_vehicle"].transform([input_vehicle])[0]
    encoded_cause = label_encoders["cause_grouped"].transform([input_cause])[0]
    encoded_weather = label_encoders["weather"].transform([input_weather])[0]
    encoded_road_shape = label_encoders["road_shape"].transform([input_road_shape])[0]
    encoded_terrain = label_encoders["terrain"].transform([input_terrain])[0]

    day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                 "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_num = day_order[input_day]

    input_data = pd.DataFrame([{
        "first_vehicle": encoded_vehicle,
        "cause_grouped": encoded_cause,
        "weather": encoded_weather,
        "road_shape": encoded_road_shape,
        "terrain": encoded_terrain,
        "hour_sin": np.sin(2 * np.pi * input_hour / 24),
        "hour_cos": np.cos(2 * np.pi * input_hour / 24),
        "day_sin": np.sin(2 * np.pi * day_num / 7),
        "day_cos": np.cos(2 * np.pi * day_num / 7),
        "month_sin": np.sin(2 * np.pi * input_month),
        "month_cos": np.cos(2 * np.pi * input_month),
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    st.subheader(f"ผลการทำนายระดับความเสี่ยง **{prediction}**")
    st.write(f"ความมั่นใจของโมเดล: **{confidence:.1f}%**")

    if prediction == "สูง":
        st.error("⚠️ เงื่อนไขนี้มีความเสี่ยงสูงที่จะเกิดอุบัติเหตุรุนแรง โปรดขับขี่ด้วยความระมัดระวังเป็นพิเศษ")
    elif prediction == "กลาง":
        st.warning("เงื่อนไขนี้มีความเสี่ยงระดับปานกลาง โปรดขับขี่ด้วยความระมัดระวัง")
    else:
        st.success("เงื่อนไขนี้มีความเสี่ยงค่อนข้างต่ำ แต่ยังต้องขับขี่อย่างระมัดระวังเสมอ")

    st.header("⚖️ ถาม-ตอบกฎหมายจราจร")
st.write("พิมพ์คำถามเกี่ยวกับกฎหมายจราจรไทย ระบบจะค้นหาและตอบจากคู่มือกฎหมายจริง")

law_question = st.text_input("คำถามของคุณ", placeholder="เช่น ขับรถเร็วเกินกำหนดมีโทษอย่างไร")

if st.button("ค้นหาคำตอบ", type="primary"):
    if law_question:
        with st.spinner("กำลังค้นหาข้อมูล..."):
            answer, sources = answer_law_question(law_question)
        
        st.write(answer)
        
        with st.expander("ดูเนื้อหาต้นฉบับที่ใช้อ้างอิง"):
            for chunk in sources:
                st.write(f"**หน้า {chunk['page']}**")
                st.text(chunk['text'][:500] + "...")
    else:
        st.warning("กรุณาพิมพ์คำถามก่อนค้นหา")