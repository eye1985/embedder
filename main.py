import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_query("This is a sample text")
    print(vectors)


if __name__ == "__main__":
    main()
