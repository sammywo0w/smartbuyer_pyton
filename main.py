from fastapi import FastAPI, HTTPException, Request
import openai
import os
import json
from dotenv import load_dotenv
from supabase import create_client

# Загрузка переменных среды
load_dotenv()

# Ключи и инициализация клиентов
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()


@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        # Получение тела запроса (работает в любом случае — даже если тело пришло строкой)
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8")
        print("📦 Raw body:", body_str)

        # Парсинг JSON
        input_data = json.loads(body_str)
        print("🔥 Supabase payload (parsed):", input_data)

        # Извлечение данных
        text = input_data.get("text", "")
        profile_id = input_data.get("id")

        # Валидация
        if not profile_id or not text:
            raise ValueError(f"Missing 'id' or 'text'. Got id: {profile_id}, text: {text}")

        # Получение эмбеддинга от OpenAI
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        print("✅ Embedding generated successfully")

        # Обновление Supabase
        supabase.table("expert_profile").update({"embedding": embedding}).eq("_id", profile_id).execute()

        return {"status": "success", "_id": profile_id}

    except Exception as e:
        print("❌ EXCEPTION:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
