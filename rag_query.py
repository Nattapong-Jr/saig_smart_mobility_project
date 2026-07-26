from google import genai
import json
import os
import numpy as np

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

with open("law_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

law_embeddings = np.load("law_embeddings.npy")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_law(question, top_k=3):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    question_embedding = np.array(result.embeddings[0].values)

    similarities = [cosine_similarity(question_embedding, emb) for emb in law_embeddings]
    top_indice = np.argsort(similarities)[::-1][:top_k]

    return [chunks[i] for i in top_indice]

def answer_question(question):
    relevant_chunks = search_law(question)

    context = "\n\n".join([f"[หน้า {c['page']}]\n{c['text']}" for c in relevant_chunks])

    prompt = f"""คุณคือผู้ช่วยตอบคำถามเกี่ยวกับกฎหมายจราจรไทย ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม ถ้าข้อมูลไม่พอตอบ ให้บอกตรงๆ
ว่าไม่พบข้อมูลที่เกี่ยวข้อง อย่าเดาเอง
    
ข้อมูลอ้างอิงซ
{context}

 คำถาม: {question}

 ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย พร้อมระบุว่าอ้างอิงจากหน้าไหน"""

    response = client.models.generate_content(
        model = "gemini-flash-latest",
        contents=prompt
    ) 

    return response.text, relevant_chunks

if __name__ == "__main__":
    question = "ขับรถเร็วเกินกำหนดมีโทษอย่างไร"
    answer, sources = answer_question(question)

    print("คำถาม:", question)
    print("\nคำตอบ")
    print(answer)
    print("\nอ้างอิงจากหน้า:", [c["page"] for c in sources])
