from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Ключи
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Клиент Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Модель запроса
class TextInput(BaseModel):
    text: str

# Правильно исправленная ручка /embed
@app.post("/embed")
async def get_embedding(input: TextInput):
    try:
        response = openai.embeddings.create(
            input=input.text,
            model="text-embedding-3-small"
        )

        embedding = response.data[0].embedding  # <-- Правильный доступ!

        return {"embedding": embedding}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Правильно исправленная ручка для webhook от Supabase
@app.post("/embed-hook")
async def embed_hook(input: dict):
    try:
        text = input.get("text", "")
        profile_id = input.get("id")

        if not profile_id or not text:
            raise ValueError("Missing 'id' or 'text'.")

        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        embedding = response.data[0].embedding  # <-- Правильный доступ!

        supabase.table("expert_profile").update({"embedding": embedding}).eq("_id", profile_id).execute()

        return {"status": "updated", "id": profile_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
