from fastapi import FastAPI, HTTPException, Request
import openai
import os
from dotenv import load_dotenv
from supabase import create_client
import json

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

        # Извлекаем нужные данные из JSON
        record = input_data.get('record', {})
        profile_id = record.get('_id')
        
        # Получаем текстовые поля для embedding
        about_me_text = record.get('about_me_text', '')
        keyachievementssuccesses_text = record.get('keyachievementssuccesses_text', '')
        current_role_text = record.get('current_role_text', '')
        
        # Собираем текст для отправки в OpenAI
        text = f"{about_me_text} {keyachievementssuccesses_text} {current_role_text}"
        
        if not profile_id or not text.strip():
            raise HTTPException(status_code=400, detail="Missing 'id' or 'text'. Make sure both are provided.")

        # Запрос к OpenAI для получения embedding
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # или другой нужный вам вариант модели
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
