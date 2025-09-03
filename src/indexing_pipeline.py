from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore




def indexing(document_store:InMemoryDocumentStore) :



    # 2. Build the Indexing Pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=3))
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    # 3. Connect the components
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    return indexing_pipeline




