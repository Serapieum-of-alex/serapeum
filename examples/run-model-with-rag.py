import os
from dotenv import load_dotenv
import torch

load_dotenv(f".env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
rdir = r"\\MYCLOUDEX2ULTRA\research\llm"
model_dir = rf"{rdir}\models"
dataset_dir = rf"{rdir}\datasets"
from transformers import (
    RagTokenizer,
    RagRetriever,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    RagTokenForGeneration,
)
from datasets import Dataset

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %% Step 1: Prepare your dataset
dataset_path = f"{dataset_dir}/trial/trial-dataset"
index_path = f"{dataset_dir}/trial/trial-index"
data = {
    "title": ["Title 1", "Title 2", "Title 3"],
    "text": [
        "Apple is looking at buying a U.K. startup for $1 billion.",
        "San Francisco considers banning sidewalk delivery robots.",
        "London is a big city in the United Kingdom.",
    ],
}

dataset = Dataset.from_dict(data)

# Create embeddings
dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base", cache_dir=model_dir
)
dpr_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base", cache_dir=model_dir
)


def embed_texts(batch):
    inputs = dpr_tokenizer(
        batch["text"], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        embeddings = dpr_encoder(**inputs).pooler_output
    batch["embeddings"] = embeddings.numpy()
    return batch


# Apply embedding function to the dataset
dataset = dataset.map(embed_texts, batched=True, batch_size=8)
#
# Verify the columns
print(dataset.column_names)
# dataset["embeddings"]
# Add FAISS index to the dataset
dataset.add_faiss_index(column="embeddings")

# %%
# Save the index to disk
dataset.get_index("embeddings").save(index_path)
# Save dataset and index to disk
dataset.drop_index(index_name="embeddings")
dataset.save_to_disk(dataset_path)

# Load the dataset and index from disk
dataset = Dataset.load_from_disk(dataset_path)
dataset.load_faiss_index("embeddings", index_path)
# %% Step 2: Load tokenizer and retriever
# Initialize RagRetriever with custom dataset and index
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
)

# Load the RAG model
model = RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-nq", cache_dir=model_dir
)

# Load the tokenizer for RAG
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq", cache_dir=model_dir)

# %%
# Tokenize input
input_text = "Tell me about Apple"
inputs = tokenizer([input_text], return_tensors="pt")

# Compute question hidden states using the DPR question encoder
with torch.no_grad():
    question_hidden_states = dpr_encoder(**inputs).pooler_output

question_hidden_states = question_hidden_states.numpy()

# Retrieve documents
retrieved_docs = retriever(
    question_input_ids=inputs["input_ids"],
    question_hidden_states=question_hidden_states,
)

# Verify retrieved documents
print("Retrieved Docs:", retrieved_docs)

# Ensure doc_scores is not None
if "doc_scores" not in retrieved_docs or retrieved_docs["doc_scores"] is None:
    # Calculate doc_scores manually
    retrieved_doc_embeds = torch.tensor(
        retrieved_docs["retrieved_doc_embeds"], dtype=torch.float
    )
    doc_scores = torch.matmul(
        torch.tensor(question_hidden_states, dtype=torch.float),
        retrieved_doc_embeds.transpose(-1, -2),
    ).squeeze(1)
else:
    doc_scores = torch.tensor(retrieved_docs["doc_scores"])

# Convert retrieved documents to tensors
context_input_ids = torch.tensor(retrieved_docs["context_input_ids"])
context_attention_mask = torch.tensor(retrieved_docs["context_attention_mask"])

# Generate answer
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        context_input_ids=context_input_ids,
        context_attention_mask=context_attention_mask,
        doc_scores=doc_scores,  # Pass the document scores to the generation method
    )

# Decode the generated answer
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text)
