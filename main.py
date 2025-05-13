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
        # Получаем данные из запроса
        input_data = await request.json()
        print("🔥 Supabase payload:", input_data)

        record = input_data.get('record', {})
        about_me_text = record.get('about_me_text', '')
        keyachievementssuccesses_text = record.get('keyachievementssuccesses_text', '')
        current_role_text = record.get('current_role_text', '')

        # Формируем текст для embedding
        combined_text = f"{about_me_text} {keyachievementssuccesses_text} {current_role_text}"

        # Получаем embedding от OpenAI
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # Новый ID модели для embedding
            input=combined_text
        )

        # Извлекаем embedding
        embedding = response['data'][0]['embedding']
        print("✅ Embedding generated successfully")

        # Обновляем данные в Supabase
        profile_id = record.get('_id')
        if profile_id:
            supabase.table("expert_profile").update({"embedding": embedding}).eq("_id", profile_id).execute()

        return {"status": "success", "_id": profile_id}

    except Exception as e:
        print("❌ EXCEPTION:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
