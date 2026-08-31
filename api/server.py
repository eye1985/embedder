from operator import itemgetter

from fastapi import FastAPI
from pydantic import BaseModel

from llm.init import init_llm

app = FastAPI()

run, config = itemgetter("runnable_with_history", "config")(init_llm())


@app.get("/")
def root():
    return {"message": "Hello World"}


class Chat(BaseModel):
    prompt: str


@app.post("/chat")
def chat(prompt: Chat):
    res = run.invoke({"question": prompt.prompt}, config=config)
    return {"message": res}
