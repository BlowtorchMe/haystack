import unittest

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.indexing_pipeline import indexing
from src.ranking_pipeline import ranking


class TestPipelines(unittest.TestCase):

    def test_indexing_and_ranking(self):
        document_store = InMemoryDocumentStore();
        indexing_pipeline = indexing(document_store=document_store)
        docs = [
            Document(content="Paris is the capital and most populous city of France."),
            Document(content="The Eiffel Tower is a famous landmark in Paris."),
            Document(content="The Louvre Museum is located in Paris, France.")
        ]
        question = "What's in Paris?"
        response = indexing_pipeline.run(data={"splitter": {"documents": docs}})
        assert response["writer"]["documents_written"] == 3


        ranking_pipeline = ranking(document_store=document_store)
        question = "What's in Paris?"
        response = ranking_pipeline.run(data={"text_embedder": {"text": question}, "bm25_retriever": {"query": question}})
