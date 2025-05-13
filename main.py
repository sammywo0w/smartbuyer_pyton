from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/embed-hook")
async def embed_hook(request: Request):
    try:
        # Просто выведем всё, что приходит
        body_bytes = await request.body()
        print("📦 Raw bytes:", body_bytes)

        body_str = body_bytes.decode("utf-8")
        print("📜 Decoded string:", body_str)

        try:
            data = json.loads(body_str)
            print("🔥 Parsed JSON:", data)
        except Exception as json_err:
            print("❌ JSON decode error:", json_err)
            data = {}

        return {"status": "ok", "raw": body_str, "parsed": data}

    except Exception as e:
        print("❌ General exception:", str(e))
        return {"error": str(e)}
