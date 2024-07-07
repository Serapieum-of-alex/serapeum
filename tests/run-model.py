import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv(f".env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
from llm.model import ChatModel, list_models

# %%
models = list_models(limit=10, author="openai")
# %%
model_dir = "local-model"
model_id = "google/gemma-2b-it"
model = ChatModel(
    model_id=model_id, device="cuda", access_token=ACCESS_TOKEN, model_dir=model_dir
)
# %% check if cuda available
print(torch.cuda.is_available())
print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
print("Current CUDA device: ", torch.cuda.current_device())
print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
# %%
print(model.generate(question="How are you?", context=""))
"""
I am an AI language model, and I do not have personal experiences or feelings. I am designed to assist and provide
information based on the knowledge I have been trained on.
I am functioning well and am here to assist you with any questions or tasks you may have. How can I help you today?
"""
print(model.generate(question="what is cmake", context=""))
print(model.generate(question="what is deltares in the Netherlands?", context=""))
# %% Document Loader and Text Splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

paper_name = rf"wf45t6000a_series.pdf"
# load PDFs
loaders = [
    PyPDFLoader(paper_name),
]
pages = []
for loader in loaders:
    pages.extend(loader.load())

"""
LangChain recommends the RecursiveCharacterTextSplitter for generic text splitting because it tries to keep text
paragraphs, sentences, and words together in one chunk.
"""
# By using the function from_huggingface_tokenizer() we define that the length of our chunk size is measured by the
# number of tokens from our encoder model.

# split text to chunks
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
    chunk_size=256,
    chunk_overlap=32,
    strip_whitespace=True,
)

docs = text_splitter.split_documents(pages)
# %% Let’s get some intuition about the chunk size and the chunk overlap:
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

chunk_size = 1
chunk_overlap = 0
print(text_splitter.split_text(text))
# ['L', 'o', 'r', 'e', 'm', ' ', 'i', 'p', 's', 'u', 'm', ' ', 'd', 'o', 'l', 'o', 'r', ' ', 's', 'i', 't', ' ',
# 'a', 'm', 'e', 't', ',', ' ', 'c', 'o', 'n', 's', 'e', 'c', 't', 'e', 't', 'u', 'r', ' ', 'a', 'd', 'i', 'p', 'i',
# 's', 'c', 'i', 'n', 'g', ' ', 'e', 'l', 'i', 't', '.']

chunk_size = 10
chunk_overlap = 0
print(text_splitter.split_text(text))
# ['Lorem ipsum', 'dolor sit', 'amet,', 'consectetur', 'adipiscing', 'elit.']

chunk_size = 50
chunk_overlap = 0
print(text_splitter.split_text(text))
# ['Lorem ipsum dolor sit amet, consectetur adipiscing', 'elit.']

chunk_size = 20
chunk_overlap = 10
print(text_splitter.split_text(text))
# ['Lorem ipsum dolor', 'ipsum dolor sit', 'dolor sit amet,', 'sit amet, consectetur', 'consectetur adipiscing',
# 'adipiscing elit.']
# %% Vector Database
"""
Faiss is a vector database library from Meta’s fundamental AI research team for efficient similarity search and
clustering of dense vectors. Using LangChain’s community integration, we can use our docs variable from the text
splitter to create a Faiss database in RAM.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

faiss_db = FAISS.from_documents(
    docs, encoder, distance_strategy=DistanceStrategy.COSINE
)

retrieved_docs = faiss_db.similarity_search("100 year return period ", k=5)
# %% User Interface with Streamlit
