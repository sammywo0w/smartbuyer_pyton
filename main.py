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

# Инициализация клиента Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        # Получаем входные данные от Supabase Webhook
        input_data = await request.json()
        print("🔥 Supabase payload:", input_data)

        record = input_data.get('record', {})
        old_record = input_data.get('old_record', {})

        # Идентификатор профиля
        profile_id = record.get('_id')
        if not profile_id:
            raise ValueError("Missing '_id' in payload")

        # Проверяем, изменились ли нужные поля
        fields_to_watch = [
            "about_me_text",
            "keyachievementssuccesses_text",
            "current_role_text",
            "searchfield",
            "suppliers_choise",
            "spec_areas_choise"
        ]

        def fields_updated(new, old, keys):
            return any(new.get(k) != old.get(k) for k in keys)

        if not fields_updated(record, old_record, fields_to_watch):
            return {"message": "No relevant fields changed. Skipping embedding update."}

        # Собираем текст для embedding
        combined_text = " ".join([
            record.get("about_me_text", ""),
            record.get("keyachievementssuccesses_text", ""),
            record.get("current_role_text", ""),
            record.get("searchfield", ""),
            record.get("suppliers_choise", ""),
            record.get("spec_areas_choise", "")
        ])

        # Получаем embedding через OpenAI
        response = openai.Embedding.create(
            model="text-embedding-3-small",  # Можно сменить на нужную модель
            input=combined_text
        )
        embedding = response['data'][0]['embedding']
        print("✅ Embedding generated successfully")

        # Обновляем или вставляем embedding в отдельную таблицу
        supabase.table("expert_embedding").upsert({
            "_id": profile_id,
            "embedding": embedding
        }).execute()

        return {"status": "success", "_id": profile_id}

    except Exception as e:
        print("❌ Exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

