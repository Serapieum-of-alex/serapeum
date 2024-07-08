from langchain_community.embeddings import HuggingFaceEmbeddings

class Encoder:

    def __init__(
            self, model_id: str = "sentence-transformers/all-MiniLM-L12-v2", device: str = "cpu",
            model_dir: str = None,
    ):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_id,
            cache_folder=model_dir,
            model_kwargs={"device": device},
        )


