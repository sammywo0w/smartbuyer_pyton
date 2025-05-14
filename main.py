from fastapi import FastAPI, HTTPException, Request, Query
from typing import List
import openai
import os
from dotenv import load_dotenv
from supabase import create_client

# Загрузка переменных среды
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Инициализация клиента Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Утилита: безопасное преобразование в строку
def safe_str(val):
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x is not None)
    return str(val) if val is not None else ""

# Утилита: проверка изменений нужных полей
def fields_updated(new, old, keys):
    if not isinstance(new, dict) or not isinstance(old, dict):
        return True  # Если вдруг нет данных — лучше перестраховаться
    return any(new.get(k) != old.get(k) for k in keys)

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        # Получение данных из webhook-запроса
        input_data = await request.json()
        print("🔥 Supabase payload:", input_data)

        record = input_data.get('record') or {}
        old_record = input_data.get('old_record') or {}
        profile_id = record.get('_id')

        if not profile_id:
            raise ValueError("Missing '_id' in record")

        # Определим, откуда пришёл webhook — по полям hourlies
        is_hourly = bool(record.get("title")) and bool(record.get("topics_text"))

        if is_hourly:
            print("💼 Обработка hourlies")

            fields_to_watch = [
                "title",
                "topics_text",
                "experience_b",
                "hourly_overvi",
                "suppliers_list",
                "search_field"
            ]

            if not fields_updated(record, old_record, fields_to_watch):
                print("ℹ️ Hourly: no relevant fields changed. Skipping update.")
                return {"message": "No relevant fields changed (hourlies)."}

            combined_text = " ".join([
                safe_str(record.get("title")),
                safe_str(record.get("topics_text")),
                safe_str(record.get("experience_b")),
                safe_str(record.get("hourly_overvi")),
                safe_str(record.get("suppliers_list")),
                safe_str(record.get("search_field")),
            ])

        else:
            print("👤 Обработка expert_profile")

            fields_to_watch = [
                "about_me_text",
                "keyachievementssuccesses_text",
                "current_role_text",
                "searchfield",
                "suppliers_choise",
                "spec_areas_choise"
            ]

            if not fields_updated(record, old_record, fields_to_watch):
                print("ℹ️ Expert: no relevant fields changed. Skipping update.")
                return {"message": "No relevant fields changed (expert)."}

            combined_text = " ".join([
                safe_str(record.get("about_me_text")),
                safe_str(record.get("keyachievementssuccesses_text")),
                safe_str(record.get("current_role_text")),
                safe_str(record.get("searchfield")),
                safe_str(record.get("suppliers_choise")),
                safe_str(record.get("spec_areas_choise")),
            ])

        # Получение embedding от OpenAI
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=combined_text
        )
        embedding = response["data"][0]["embedding"]
        print("✅ Embedding generated")

        # Сборка записи для сохранения
        embedding_record = {
            "_id": profile_id,
            "embedding": embedding
        }

        if is_hourly:
            embedding_record["hourlie_id"] = record.get("id_hourly")

        supabase.table("expert_embedding").upsert(embedding_record).execute()
        print("✅ Embedding saved to expert_embedding")

        return {"status": "success", "_id": profile_id}

    except Exception as e:
        print("❌ Exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_similar_profiles(request: Request, top_k: int = Query(default=5)):
    try:
        data = await request.json()
        query_text = data.get("query", "")

        if not query_text:
            raise ValueError("Query is empty")

        # Получаем embedding от OpenAI
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_embedding = response['data'][0]['embedding']

        # Вызываем RPC-функцию в Supabase
        result = supabase.rpc("search_embeddings", {
            "query_embedding": query_embedding,
            "top_k": top_k
        }).execute()

        matches = result.data if result else []
        return {"results": matches}

    except Exception as e:
        print("❌ Search exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
