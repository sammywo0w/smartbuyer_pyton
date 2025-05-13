from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(input: TextInput):
    try:
        response = openai.Embedding.create(
            input=input.text,
            model="text-embedding-3-small"
        )
        return {"embedding": response['data'][0]['embedding']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
        embedding = response['data'][0]['embedding']

        supabase.table("expert_profile").update({"embedding": embedding}).eq("id", profile_id).execute()

        return {"status": "updated", "id": profile_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
