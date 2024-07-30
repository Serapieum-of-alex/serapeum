import os
from dotenv import load_dotenv

load_dotenv(f".env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
from serapeum.chat_model import ChatModel, list_models

model_dir = r"C:\MyComputer\llm\models"
# model_dir = r"\\MYCLOUDEX2ULTRA\research\llm\models"
# %%
models = list_models(limit=10, author="openai")
# %%
chat_template_path = "llama-3-instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = ChatModel(
    model_id=model_id, device="cuda", access_token=ACCESS_TOKEN, model_dir=model_dir
)
tokenizer = model.tokenizer
# %%
# import torch
# max_new_tokens = 250
# device = "cuda"
#
# question = "What is the capital of France?"
# messages = [
#     {"role": "system", "content": "You are a knowledgeable assistant who helps users with their questions."},
#     {"role": "user", "content": f"{question}"},
#     {"role": "assistant", "content": "You are a helpful assistant."}
#
# ]
# formatted_prompt = tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True, padding=True
# )
#
# inputs = tokenizer.encode(
#     formatted_prompt, add_special_tokens=False, return_tensors="pt"
# ).to(device)
# # pad_token_id=self.tokenizer.eos_token_id,
# with torch.no_grad():
#     outputs = model._model.generate(
#         input_ids=inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#     )
# response = tokenizer.decode(outputs[0], skip_special_tokens=False)
# print(response)
# response = response[len(formatted_prompt):]
# # remove eos token
# response = response.replace("<eos>", "")
# %%
print(model.generate(question="How are you?", context=""))
"""
I am an AI language model, and I do not have personal experiences or feelings. I am designed to assist and provide
information based on the knowledge I have been trained on.
I am functioning well and am here to assist you with any questions or tasks you may have. How can I help you today?
"""
print(model.generate(question="what is cmake", context=""))
# %%
con = (
    """
Deltares focuses on the Dutch, European and global missions. It is the impact of these agendas that determine our
priorities and actions. This mission-driven way of working is given focus at Deltares with so-called 'moonshots',
ambitious goals with great social importance. We can only achieve this by working together with other (knowledge)
partners and our clients.
"""
    * 100
)
response = model.generate(question="what is deltares in the Netherlands?", context=con)
print(response)
# %% Document Loader and Text Splitter
from serapeum.datasource import DataSource

# paper_name = ["tests/data/wf45t6000a_series.pdf"]
paper_name = ["tests/data/paper.pdf"]
datasource = DataSource(dtype="pdf", file_paths=paper_name)

model_id = "sentence-transformers/all-MiniLM-L12-v2"
datasource.create_splitter(model_id=model_id, chunk_size=256, overlap=25)
# paper_name = ["paper.pdf"]
docs = datasource.split_data()
# %% Vector Database
"""
Faiss is a vector database library from Meta’s fundamental AI research team for efficient similarity search and
clustering of dense vectors. Using LangChain’s community integration, we can use our docs variable from the text
splitter to create a Faiss database in RAM.
"""
from serapeum.encoder import Encoder

encoder = Encoder(
    model_id="sentence-transformers/all-MiniLM-L12-v2",
    device="cuda",
    model_dir=model_dir,
)
from serapeum.datastore import Faiss

DB = Faiss(docs, encoder.embedding_function)
user_prompt = "how to select a location"
retrieved_docs = DB.similarity_search(user_prompt, k=5)
# retrieved_docs = DB.similarity_search("how do i drain the filter", k=10)
len(retrieved_docs)
# %%
user_prompt = "what is a return period"
retrieved_docs = """A return period, also known as a recurrence interval or repeat interval, is an average time or an
estimated average time between events such as earthquakes, floods,[1] landslides,[2] or river discharge flows to occur.
It is a statistical measurement typically based on historic data over an extended period, and is used usually for risk analysis. Examples include deciding whether a project should be allowed to go forward in a zone of a certain risk or designing structures to withstand events with a certain return period. The following analysis assumes that the probability of the event occurring does not vary over time and is independent of past events."""
# %%
answer = model.generate(user_prompt, context=retrieved_docs, max_new_tokens=512)

# %%
user_prompt = "what is a return period"
answer = model.generate(user_prompt, context=retrieved_docs, max_new_tokens=512)
