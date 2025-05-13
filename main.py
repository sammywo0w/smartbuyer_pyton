from fastapi import FastAPI, Request
import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        body = await request.json()
        print("🔥 Parsed JSON:", body)

        # Извлечение необходимых полей из JSON
        record = body.get('record', {})
        about_me_text = record.get('about_me_text', '')
        keyachievementssuccesses_text = record.get('keyachievementssuccesses_text', '')
        current_role_text = record.get('current_role_text', '')

        # Конкатенация всех нужных текстов для создания embedding
        full_text = f"{about_me_text} {keyachievementssuccesses_text} {current_role_text}"

        # Генерация embedding
        response = openai.Embedding.create(
            input=full_text,
            model="text-embedding-ada-002"
        )

        embedding = response['data'][0]['embedding']
        print("✅ Embedding generated successfully")

        # Здесь можно сохранить embedding в базе данных или передать куда-то еще

        return {"status": "success", "embedding": embedding}

    except Exception as e:
        print("❌ EXCEPTION:", str(e))
        return {"error": str(e)}
