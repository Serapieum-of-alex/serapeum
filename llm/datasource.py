"""
Data source
"""

from typing import List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


class DataSource:
    """
    DataSource class for loading and splitting text data from various sources.
    """

    def __init__(
        self,
        dtype: str = "pdf",
        file_paths: List[str] = None,
    ):
        """DataSource constructor

        Parameters
        ----------
        dtype: str, default is pdf.
            it can be "pdf", "csv", "json", etc.
        file_paths: str/List[str]
            the path to one file, or a list of strings.

        Examples
        --------
        >>> datasource = DataSource(dtype="pdf", file_paths=["tests/data/test-pdf.pdf"])
        >>> print(datasource)
        <BLANKLINE>
                Type: pdf
                Number of Pages: 40
        <BLANKLINE>
        >>> print(datasource.data) # doctest: +SKIP
        [
            Document(
                metadata={'source': 'tests/data/test-pdf.pdf', 'page': 0},
                page_content='Washer User manual WF45T6000A* Untitled-31   1 2020-03-02   PM 3:19:32'),
            Document(
                metadata={'source': 'tests/data/test-pdf.pdf', 'page': 1},
                page_content='......')
                ]
        """
        self.dtype = dtype
        loaders = [PyPDFLoader(file_path) for file_path in file_paths]
        pages = []

        for loader in loaders:
            pages.extend(loader.load())

        self._data = pages

    def __str__(self):
        message = f"""
        Type: {self.dtype}
        Number of Pages: {len(self.data)}
        """
        return message

    @property
    def data(self) -> List[Document]:
        """Data"""
        return self._data

    def create_splitter(
        self, model_id: str = None, chunk_size: int = 256, overlap: int = None
    ):
        """Text splitter.

        Parameters
        ----------
        model_id: str, default is None
            i.e. "sentence-transformers/all-MiniLM-L12-v2".
        chunk_size: int, default is 256.
            length of the chuck.
        overlap: int, default is None.
            the overlap between two consecutive chunks. f None, the overlap is set to 10% of the chunk_size.

        Examples
        --------
        >>> my_data = DataSource(dtype="pdf", file_paths=["tests/data/test-pdf.pdf"])
        >>> my_data.create_splitter(model_id="sentence-transformers/all-MiniLM-L12-v2")
        >>> print(my_data.splitter)
        <langchain_text_splitters.character.RecursiveCharacterTextSplitter object at 0x0000028E8AE59DF0>
        """
        # number of tokens per chunk (for text-based data types)
        chunk_size = chunk_size
        overlap = int(chunk_size / 10) if overlap is None else overlap
        if model_id is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                strip_whitespace=True,
            )
        else:
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=AutoTokenizer.from_pretrained(model_id),
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                strip_whitespace=True,
            )
        self._splitter = splitter

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        """Text splitter"""
        return self._splitter

    def split_data(self) -> List[Document]:
        """chunk pdf

        Returns
        -------
        List[Document]
            A list of langchain core documents, based on the chunk size and the overlap; the main pdf will be chunked
            into smaller chunks.

        Examples
        --------
        >>> my_data = DataSource(dtype="pdf", file_paths=["tests/data/test-pdf.pdf"])
        >>> my_data.create_splitter(model_id="sentence-transformers/all-MiniLM-L12-v2")
        >>> data_chunk = my_data.split_data()
        >>> print(data_chunk) # doctest: +SKIP
        [
            Document(
                metadata={'source': 'tests/data/test-pdf.pdf', 'page': 0},
                page_content='chunk 1'),
            Document(
                metadata={'source': 'tests/data/test-pdf.pdf', 'page': 1},
                page_content='chunk 2',
            Document(
                metadata={'source': 'tests/data/test-pdf.pdf', 'page': 1},
                page_content='chunk 3'),
            Document(
                metadata={'source': 'tests/data/test-pdf.pdf', 'page': 1},
                page_content='chunk 4'),
            ]
        """
        docs = self.splitter.split_documents(self.data)
        return docs
