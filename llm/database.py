"""
Data Base class
"""
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class Faiss:
    """
    Faiss class for managing and querying FAISS-based vector databases.
    """

    def __init__(self, docs, embedding_function):
        self.db = FAISS.from_documents(
            docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        )

    def similarity_search(self, question: str, k: int = 5) -> str:
        """Similarity search.

        Parameters
        ----------
        question: str
            text search.
        k: int
            number of most similar documents to return.

        Returns
        -------
        context: str
            the most similar documents joined as one string
        """
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
