from transformers import AutoTokenizer
model_dir = r"\\MYCLOUDEX2ULTRA\research\llm"
# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased", model_dir=model_dir
)

# Example long text
text = "This is a long text that needs to be chunked. " * 100  # Simulating a long document

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Define chunk size and overlap
chunk_size = 512
chunk_overlap = 50

# Create chunks
chunks = []
for i in range(0, len(tokens), chunk_size - chunk_overlap):
    chunk = tokens[i:i + chunk_size]
    chunks.append(chunk)

# Convert chunks back to token IDs
input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in chunks]

# Print the number of chunks and the size of the first chunk
print(f"Number of chunks: {len(chunks)}")
print(f"Size of the first chunk: {len(chunks[0])} tokens")
#%%
# %% Let’s get some intuition about the chunk size and the chunk overlap:
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
    chunk_size=256,
    chunk_overlap=32,
    strip_whitespace=True,
)

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
