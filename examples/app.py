import os
import streamlit as st
from serapeum.chat_model import ChatModel
from serapeum.encoder import Encoder
from serapeum.datasource import DataSource
from serapeum.datastore import Faiss

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

model_dir = r"\\MYCLOUDEX2ULTRA\research\llm"
st.title("LLM Chatbot RAG Assistant")


@st.cache_resource
def load_model():
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda", model_dir=model_dir)
    return model


@st.cache_resource
def load_encoder():
    encoder = Encoder(
        model_id="sentence-transformers/all-MiniLM-L12-v2",
        device="cuda",
        model_dir=model_dir,
    )
    return encoder


model = load_model()  # load our models once and then cache it
encoder = load_encoder()


def save_file(uploaded_file):
    """helper function to save documents to disk"""
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


with st.sidebar:
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
    k = st.number_input("k", 1, 10, 3)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
    )
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))

    if uploaded_files:
        datasource = DataSource(dtype="pdf", file_paths=file_paths)
        datasource.create_splitter(
            model_id="sentence-transformers/all-MiniLM-L12-v2",
            chunk_size=256,
            overlap=25,
        )
        docs = datasource.split_data()
        DB = Faiss(docs=docs, embedding_function=encoder.embedding_function)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Add a user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display a user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        context = (
            None if uploaded_files == [] else DB.similarity_search(user_prompt, k=k)
        )
        answer = model.generate(
            user_prompt, context=context, max_new_tokens=max_new_tokens
        )
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
