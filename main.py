from fastapi import FastAPI, HTTPException, Request
import openai
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Загрузка переменных среды
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Клиент Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        input_data = await request.json()
        print("🔥 Supabase payload:", input_data)

        # Извлекаем нужные данные из запроса
        text = input_data.get("text", "")
        profile_id = input_data.get("id")

        if not profile_id or not text:
            raise ValueError(f"Missing 'id' or 'text'. Got id: {profile_id}, text: {text}")

        # Запрос к OpenAI для получения embedding
        response = openai.embeddings.create(
            model="text-embedding-ada-002",  # Можно выбрать подходящую модель
            input=text
        )

        embedding = response['data'][0]['embedding']
        print("✅ Embedding generated successfully")

        # Обновление записи в базе данных Supabase
        supabase.table("expert_profile").update({"embedding": embedding}).eq("_id", profile_id).execute()

        return {"status": "success", "_id": profile_id}

    except Exception as e:
        print("❌ EXCEPTION:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
