import hashlib
from webbrowser import Error
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json

CACHE_FILE = "cache.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExplanationRequest(BaseModel):
    original_text: str
    user_hobby: str
    lesson_context: str

def load_cache():
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def generate_md5(request: ExplanationRequest):
    hash_input = f"{request.original_text}|{request.user_hobby}|{request.lesson_context}"
    return hashlib.md5(hash_input.encode()).hexdigest()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)


@app.post("/generate_explanation")
async def generate_explanation(request: ExplanationRequest):
    """
    POST API to generate a hobby-themed explanation using a locally hosted Llama model with caching.
    """
    cache = load_cache()
    request_hash = generate_md5(request)

    # Check cache first
    if request_hash in cache:
        return {"response": cache[request_hash], "cached": True}

    # Prepare the input prompt
    query = (
        f"Rewrite the following explanation strictly in under 20 words in a way that relates to {request.lesson_context}.\n"
        f"with respect to user hobby: {request.user_hobby}\n"
        f"Original Text: {request.original_text}\n"
        "Make the explanation engaging and simple to understand."
    )

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct:free",
            messages=[{'role': 'user', 'content': query}]
        )
        response = completion.choices[0].message.content

        # Store response in cache
        cache[request_hash] = response
        save_cache(cache)

        return {"response": response, "cached": False}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
