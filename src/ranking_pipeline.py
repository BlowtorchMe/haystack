from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


def ranking(document_store:InMemoryDocumentStore):
    bm25_retriever = InMemoryBM25Retriever(document_store=document_store)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    text_embedder = SentenceTransformersTextEmbedder()
    document_joiner = DocumentJoiner(join_mode="concatenate")
    ranker = SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    prompt_builder = PromptBuilder("Based on the provided context, answer the question. Context: {{ documents }}. Question: {{ question }}")
    generator = OpenAIGenerator()

    query_pipeline = Pipeline()
    query_pipeline.add_component("bm25_retriever", bm25_retriever)
    query_pipeline.add_component("embedding_retriever", embedding_retriever)
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("document_joiner", document_joiner)
    query_pipeline.add_component("ranker", ranker)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("generator", generator)


    # Embed the query and send it to the embedding retriever
    query_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    # Send the raw question to BM25 retriever
    query_pipeline.connect("text_embedder.text", "bm25_retriever.query")
    # Merge results from both retrievers
    query_pipeline.connect("bm25_retriever.documents", "document_joiner.documents")
    query_pipeline.connect("embedding_retriever.documents", "document_joiner.documents")

    # Send merged docs to the ranker (along with original question)
    query_pipeline.connect("document_joiner.documents", "ranker.documents")
    query_pipeline.connect("text_embedder.text", "ranker.query")

    # Send top-ranked docs + question to the prompt builder
    query_pipeline.connect("ranker.documents", "prompt_builder.documents")
    query_pipeline.connect("text_embedder.text", "prompt_builder.question")

    # Send the prompt to the generator
    query_pipeline.connect("prompt_builder", "generator")