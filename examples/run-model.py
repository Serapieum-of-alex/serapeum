import os
from dotenv import load_dotenv
load_dotenv(f".env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
from llm.model import ChatModel, list_models
# %%
models = list_models(limit=10, author="openai")
# %%
model_dir = r"\\MYCLOUDEX2ULTRA\research\llm"
model_id = "meta-llama/Meta-Llama-3-8B"
model = ChatModel(
    model_id=model_id, device="cuda", access_token=ACCESS_TOKEN, model_dir=model_dir
)
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
encoder = Encoder(model_id="sentence-transformers/all-MiniLM-L12-v2", device="cuda", model_dir=model_dir)

datasource = DataSource(dtype="pdf", chunk_size=256, overlap=25, model="sentence-transformers/all-MiniLM-L12-v2")
# paper_name = ["paper.pdf"]
paper_name = [rf"wf45t6000a_series.pdf"]
docs = datasource.load_data(paper_name)
# %% Vector Database
"""
Faiss is a vector database library from Meta’s fundamental AI research team for efficient similarity search and
clustering of dense vectors. Using LangChain’s community integration, we can use our docs variable from the text
splitter to create a Faiss database in RAM.
"""
from llm.database import Faiss
DB = Faiss(docs, encoder.embedding_function)
user_prompt = "how to select a location"
retrieved_docs = DB.similarity_search(user_prompt, k=5)
# retrieved_docs = DB.similarity_search("how do i drain the filter", k=10)
len(retrieved_docs)
#%%
user_prompt = "what is a return period"
retrieved_docs = """A return period, also known as a recurrence interval or repeat interval, is an average time or an 
estimated average time between events such as earthquakes, floods,[1] landslides,[2] or river discharge flows to occur.
It is a statistical measurement typically based on historic data over an extended period, and is used usually for risk analysis. Examples include deciding whether a project should be allowed to go forward in a zone of a certain risk or designing structures to withstand events with a certain return period. The following analysis assumes that the probability of the event occurring does not vary over time and is independent of past events."""
#%%
answer = model.generate(user_prompt, context=retrieved_docs, max_new_tokens=512)
