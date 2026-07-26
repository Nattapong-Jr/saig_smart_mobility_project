import pdfplumber
import json
import re

def clean_text(text):
    return re.sub(r'[\uf700-\uf8ff]', '', text)

chunks = []
with pdfplumber.open("กฎหมายจราจร.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            cleaned = clean_text(text.strip())
            chunks.append({
                "page": i + 1,
                "text": cleaned
            })

print(f"จำนวน chunk ทั้งหมด: {len(chunks)}")
print(f"\nตัวอย่าง chunk แรก:")
print(chunks[0])

with open("law_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("\nบันทึกไฟล์ law_chunks.json สำเร็จแล้ว")