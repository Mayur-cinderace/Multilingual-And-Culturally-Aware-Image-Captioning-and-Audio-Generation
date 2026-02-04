# ollama_proxy.py
from fastapi import FastAPI
import requests

app = FastAPI()
OLLAMA = "http://127.0.0.1:11434"

@app.get("/api/tags")
def tags():
    return requests.get(f"{OLLAMA}/api/tags").json()

@app.post("/api/chat")
def chat(payload: dict):
    return requests.post(f"{OLLAMA}/api/chat", json=payload).json()
