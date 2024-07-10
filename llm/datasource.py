"""
Data source
"""
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
        """

        Parameters
        ----------
        dtype: str, default is pdf.
            can be "pdf", "csv", "json", etc.
        model: str, default is "sentence-transformers/all-MiniLM-L12-v2".
            model id.
        chunk_size: int, default is 256.
            length of the chuck.
        overlap: int, default is None.
            the overlap between two consecutive chunks. f None, the overlap is set to 10% of the chunk_size.

        Examples
        --------
        >>> datasource = DataSource(
        ...    dtype="pdf", chunk_size=512, overlap=25, model="sentence-transformers/all-MiniLM-L12-v2"
        ...)
        """
        self.dtype = dtype
        # number of tokens per chunk (for text-based data types)
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size/10) if overlap is None else overlap
        self.model = model

    def load_data(self, paths: Union[str, List[str]]):
        """read the data.

        Parameters
        ----------
        paths: str/List[str]
            path to one file, or a list of strings.

        Returns
        -------
        List[Document]
            A list of langchain core documents, based on the chunk size and the overlap; the main pdf will be chunked
            into smaller chunks.

        Examples
        --------
        - First create the `DataSource` object.

            >>> datasource = DataSource(
            ...    dtype="pdf", chunk_size=512, overlap=25, model="sentence-transformers/all-MiniLM-L12-v2"
            ...)

        - Then pass a list of file names to the `load_data` function.

            >>> docs = datasource.load_data(["wf45t6000a_series.pdf"])
            >>> print(docs[:2]) # doctest: +SKIP
            [Document(metadata={'source': 'wf45t6000a_series.pdf', 'page': 0}, page_content='Washer\nUser manual\nWF45T6000A*\nUntitled-31
            1 2020-03-02   PM 3:19:32'),
            Document(metadata={'source': 'wf45t6000a_series.pdf', 'page': 1}, page_content='English2
            \nContentsContents\nSafety information 4\nWhat you need to know about the safety instructions 4\nImportant
            safety symbols 4\nImportant safety precautions 5\nCALIFORNIA PROPOSITION 65 WARNING 6\nCritical installation
            warnings 6\nInstallation cautions 8\nCritical usage warnings 8\nUsage cautions 10\nCritical cleaning
            warnings 13\nInstallation 14\nUnpacking your washer 14\nWhat’s included 15\nInstallation requirements
            17\nStep-by-step installation 21\nBefore you start 26\nInitial settings 26\nLaundry guidelines 26\nDetergent
            drawer guidelines 28\nOperations 31\nControl panel 31\nSimple steps to start 34\nCycle overview 35\nCycle
            chart 36\nSpecial features 38\nUntitled-31   2 2020-03-02   PM 3:19:32')]
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
        List[Document]
            A list of langchain core documents, based on the chunk size and the overlap; the main pdf will be chunked
            into smaller chunks.
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
