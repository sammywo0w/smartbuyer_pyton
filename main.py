from fastapi import FastAPI, HTTPException, Request
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
    return any(new.get(k) != old.get(k) for k in keys)

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        # Получение данных из webhook-запроса
        input_data = await request.json()
        print("🔥 Supabase payload:", input_data)

        record = input_data.get('record', {})
        old_record = input_data.get('old_record', {})
        profile_id = record.get('_id')

        if not profile_id:
            raise ValueError("Missing '_id' in record")

        # Поля, по которым отслеживаются изменения
        fields_to_watch = [
            "about_me_text",
            "keyachievementssuccesses_text",
            "current_role_text",
            "searchfield",
            "suppliers_choise",
            "spec_areas_choise"
        ]

        if not fields_updated(record, old_record, fields_to_watch):
            print("ℹ️ No relevant fields changed. Skipping update.")
            return {"message": "No relevant fields changed."}

        # Формируем текст для embedding
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

        # Запись в таблицу expert_embedding
        supabase.table("expert_embedding").upsert({
            "_id": profile_id,
            "embedding": embedding
        }).execute()
        print("✅ Embedding saved to expert_embedding")

        return {"status": "success", "_id": profile_id}

    except Exception as e:
        print("❌ Exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import Query
from typing import List

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

        # Выполняем SQL-запрос к Supabase (pgvector оператор cosine distance <#>)
        sql = """
            SELECT _id, 1 - (embedding <#> %s) AS similarity
            FROM expert_embedding
            ORDER BY embedding <#> %s
            LIMIT %s;
        """

        # Вызов raw SQL через postgrest может быть неудобен,
        # поэтому лучше сделать это через Supabase RPC (если настроено)
        # или psycopg2 / asyncpg — но для простоты мы используем Supabase REST

        # Прямого метода RPC с raw SQL через supabase-py нет,
        # поэтому ПОДСКАЖИ, если хочешь — я помогу сделать SQL-функцию в Supabase,
        # тогда тут можно будет вызвать `.rpc("search_embeddings", { ... })`

        return {
            "message": "❗ Для поиска лучше настроить Supabase RPC или direct DB access"
        }

    except Exception as e:
        print("❌ Search exception:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

