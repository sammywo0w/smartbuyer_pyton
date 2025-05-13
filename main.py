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
