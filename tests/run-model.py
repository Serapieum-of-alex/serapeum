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
model_dir = r"\\MYCLOUDEX2ULTRA\research\llm"
model_id = "google/gemma-2b-it"
model = ChatModel(
    model_id=model_id, device="cuda", access_token=ACCESS_TOKEN, model_dir=model_dir
)
# sentence_model = ChatModel(model_id="sentence-transformers/all-mpnet-base-v2", device="cuda", model_dir=model_dir, access_token=ACCESS_TOKEN)
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
# %%
con = """
Deltares focuses on the Dutch, European and global missions. It is the impact of these agendas that determine our 
priorities and actions. This mission-driven way of working is given focus at Deltares with so-called 'moonshots', 
ambitious goals with great social importance. We can only achieve this by working together with other (knowledge) 
partners and our clients.
""" * 100
response = model.generate(question="what is deltares in the Netherlands?", context=con)
print(response)
# %% Document Loader and Text Splitter
from llm.datasource import DataSource
from llm.encoder import Encoder

datasource = DataSource(dtype="pdf", chunk_size=256, overlap=25, model="sentence-transformers/all-MiniLM-L12-v2")
paper_name = ["paper.pdf"]
# paper_name = [rf"wf45t6000a_series.pdf"]
docs = datasource.load_data(paper_name)

encoder = Encoder(model_id="sentence-transformers/all-MiniLM-L12-v2", device="cuda", model_dir=model_dir)
# %% Vector Database
"""
Faiss is a vector database library from Meta’s fundamental AI research team for efficient similarity search and
clustering of dense vectors. Using LangChain’s community integration, we can use our docs variable from the text
splitter to create a Faiss database in RAM.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

faiss_db = FAISS.from_documents(
    docs, encoder.embedding_function, distance_strategy=DistanceStrategy.COSINE
)

retrieved_docs = faiss_db.similarity_search("what does return period means", k=5)

retrieved_docs[0].page_content

