from fastapi import FastAPI, HTTPException, Request
import openai
import os
import uuid
import traceback
import ast
from dotenv import load_dotenv
from supabase import create_client

# Загрузка переменных среды
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Инициализация Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

def safe_str(val):
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x is not None)
    return str(val) if val is not None else ""

def fields_updated(new, old, keys):
    if not isinstance(new, dict) or not isinstance(old, dict):
        return True
    return any(new.get(k) != old.get(k) for k in keys)

def is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except:
        return False

def ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except:
            return []
    return []

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        input_data = await request.json()
        record = input_data.get("record") or {}
        old_record = input_data.get("old_record") or {}
        _id = record.get("_id")

        if not _id:
            raise ValueError("Missing '_id' in record")

        is_hourly = bool(record.get("title")) and bool(record.get("topics_text"))

        if is_hourly:
            fields_to_watch = [
                "title", "topics_text", "experience_benefits_delivered",
                "categories_list_custom_categories", "suppliers"
            ]
            if not fields_updated(record, old_record, fields_to_watch):
                return {"message": "No relevant fields changed (hourlies)."}

            combined_text = " ".join([
                safe_str(record.get("title")),
                safe_str(record.get("topics_text")),
                safe_str(record.get("experience_benefits_delivered")),
                safe_str(record.get("categories_list_custom_categories")),
                safe_str(record.get("suppliers")),
            ])
        else:
            fields_to_watch = [
                "about_me_text", "keyachievementssuccesses_text",
                "current_role_text", "searchfield",
                "suppliers_choise", "spec_areas_choise", "current_employer_name_text",
                "categories_list_custom_categories"
            ]
            if not fields_updated(record, old_record, fields_to_watch):
                return {"message": "No relevant fields changed (expert)."}

            user_data_result = supabase.table("user_data_bubble") \
                .select("firstname_text, lastname_text, email, user_status_option_user_status0") \
                .eq("_id", _id).execute()

            user_data = user_data_result.data[0] if user_data_result.data else {}

            combined_text = " ".join([
                safe_str(user_data.get("firstname_text")),
                safe_str(user_data.get("lastname_text")),
                safe_str(user_data.get("email")),
                safe_str(user_data.get("user_status_option_user_status0")),
                safe_str(record.get("about_me_text")),
                safe_str(record.get("keyachievementssuccesses_text")),
                safe_str(record.get("current_role_text")),
                safe_str(record.get("searchfield")),
                safe_str(record.get("suppliers_choise")),
                safe_str(record.get("spec_areas_choise")),
                safe_str(record.get("current_employer_name_text")),
            ])

        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=combined_text
        )
        embedding = response["data"][0]["embedding"]

        hourlie_id = record.get("id_hourly")
        if not is_hourly or not is_valid_uuid(str(hourlie_id)):
            hourlie_id = None

        profile_type = "hourlie" if is_hourly else "expert"

        embedding_record = {
            "_id": _id,
            "embedding": embedding,
            "category": ensure_list(record.get("categories_list_custom_categories")),
            "skills": ensure_list(record.get("suppliers_choise")),
            "badges": ensure_list(record.get("spec_areas_choise")),
            "hourlie_id": hourlie_id,
            "type": profile_type
        }

        query = supabase.table("expert_embedding") \
            .select("id_embedding") \
            .eq("_id", _id) \
            .eq("type", profile_type)

        if profile_type == "hourlie" and hourlie_id:
            query = query.eq("hourlie_id", hourlie_id)

        existing = query.execute()

        if existing.data:
            id_embedding = existing.data[0]['id_embedding']
            supabase.table("expert_embedding").update(embedding_record).eq("id_embedding", id_embedding).execute()
        else:
            supabase.table("expert_embedding").insert(embedding_record).execute()

        return {"status": "success", "updated": _id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_similar_profiles(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query", "")

        if not query_text:
            raise ValueError("Query is empty")

        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        query_embedding = response["data"][0]["embedding"]

        result = supabase.rpc("search_embeddings", {
            "query_embedding": query_embedding
        }).execute()

        matches = result.data if result else []

        return {"results": matches}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
