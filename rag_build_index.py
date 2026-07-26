from google import genai
import json
import os
import numpy as np

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

with open("law_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"กำลังสร้าง embedding สำหรับ {len(chunks)} chunks...")

embeddings = []
for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk["text"]
    )
    embeddings.append(result.embeddings[0].values)
    print(f"  สร้าง embedding หน้า {chunk['page']} เสร็จแล้ว ({i+1}/{len(chunks)})")

embeddings_array = np.array(embeddings)
np.save("law_embeddings.npy", embeddings_array)

print(f"\nบันทึก embeddings สำเร็จ ขนาด: {embeddings_array.shape}")