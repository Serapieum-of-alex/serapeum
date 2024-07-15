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
# %%
import numpy as np


q = encoder.embed_query("What is an apple?")
z1 = encoder.embed_query(
    "An apple is a round, edible fruit produced by an apple tree (Malus spp., among them the domestic or orchard apple; Malus domestica)."
)  # from wikipedia
z2 = encoder.embed_query(
    "The cat (Felis catus), commonly referred to as the domestic cat or house cat, is the only domesticated species in the family Felidae."
)  # from wikipedia

print(np.dot(q, z1) / (np.linalg.norm(q) * np.linalg.norm(z1)))

print(np.dot(q, z2) / (np.linalg.norm(q) * np.linalg.norm(z2)))
