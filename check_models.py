from google import genai
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("--- โมเดลทั้งหมดที่ใช้ได้ ---")
for model in client.models.list():
    print(model.name, "-", model.supported_actions)