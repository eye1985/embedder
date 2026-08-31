from operator import itemgetter
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from llm.init import init_llm

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI()

run, config = itemgetter("runnable_with_history", "config")(init_llm())

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


class Chat(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(prompt: Chat):
    async def stream():
        async for chunk in run.astream({"question": prompt.prompt}, config=config):
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
