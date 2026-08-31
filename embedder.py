from typing import Literal
from langchain_openai import OpenAIEmbeddings

SupportedModels = Literal["text-embedding-3-small", "text-embedding-3-large"]

def embedder(model: SupportedModels = "text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model)
    vectors = embeddings.embed_query("This is a sample text")
    return vectors
