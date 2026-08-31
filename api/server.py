from operator import itemgetter

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from llm.init import init_llm

app = FastAPI()

run, config = itemgetter("runnable_with_history", "config")(init_llm())


@app.get("/")
def root():
    return {"message": "Hello World"}


class Chat(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(prompt: Chat):
    async def stream():
        async for chunk in run.astream({"question": prompt.prompt}, config=config):
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
