from typing import List
from serapeum.datasource import DataSource
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def test_create_datasource(manual_pdf: List[str], num_manual_pages: int):
    datasource = DataSource(dtype="pdf", file_paths=manual_pdf)
    assert datasource.dtype == "pdf"
    assert isinstance(datasource.data[0], Document)
    assert len(datasource.data) == num_manual_pages
    assert isinstance(datasource.__str__(), str)


def test_create_splitter(manual_pdf: List[str]):
    datasource = DataSource(dtype="pdf", file_paths=manual_pdf)
    datasource.create_splitter(
        model_id="sentence-transformers/all-MiniLM-L12-v2", chunk_size=256, overlap=None
    )
    assert isinstance(datasource.splitter, RecursiveCharacterTextSplitter)
    splitted_data = datasource.split_data()
    assert isinstance(splitted_data[0], Document)
    assert len(splitted_data) == 277
