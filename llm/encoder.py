from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


class Encoder:
    """Encoder"""
    def __init__(
            self, model_id: str = "sentence-transformers/all-MiniLM-L12-v2", device: str = "cpu",
            model_dir: str = None,
    ):
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_id,
            cache_folder=model_dir,
            model_kwargs={"device": device},
        )

