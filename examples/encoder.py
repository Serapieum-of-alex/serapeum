import os
from dotenv import load_dotenv

load_dotenv(f".env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings


# %%
encoder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"}
)

# %%
embeddings = encoder.embed_query("How are you?")
print(embeddings)
len(embeddings)
