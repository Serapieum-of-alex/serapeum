from typing import Union, List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


class DataSource:
    """
    DataSource class for loading and splitting text data from various sources.
    """

    def __init__(
            self, dtype: str = "pdf", model: str = "sentence-transformers/all-MiniLM-L12-v2", chunk_size: int = 256,
            overlap: int = None,
    ):
        # can be "pdf", "csv", "json", etc.
        self.dtype = dtype
        # number of tokens per chunk (for text-based data types)
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size/10) if overlap is None else overlap
        self.model = model

    def load_data(self, paths: Union[str, List[str]]):
        """

        Parameters
        ----------
        paths: str/List[str]
            path to one file, or a list of strings.

        Returns
        -------

        """
        if self.dtype == "pdf":
            return self._load_and_split_pdfs(paths)
        else:
            # add other data loading logic as needed
            return []

    def _load_and_split_pdfs(self, file_paths: list) -> List[Document]:
        """chunk pdf

        Parameters
        ----------
        file_paths: str/List[str]
            path to one file, or a list of strings.

        Returns
        -------

        """
        loaders = [PyPDFLoader(file_path) for file_path in file_paths]
        pages = []

        for loader in loaders:
            pages.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.model),
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            strip_whitespace=True,
        )
        docs = text_splitter.split_documents(pages)
        return docs
