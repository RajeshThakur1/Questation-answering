from haystack.nodes import TfidfRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
# https://haystack.deepset.ai/pipeline_nodes/retriever
document_store = InMemoryDocumentStore()
retriever = TfidfRetriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
reader.save('reader') # saving reader model

sample_query = "Who inspired the author to write this book?"
params={"Retriever": {"top_k": 5}, # Top 5 relevant documents in document_store
        "Reader": {"top_k": 3} # Top 3 answers, searched in retrieved documents.
       }
prediction = ExtractiveQAPipeline.run(query=sample_query, params=params)
print(prediction['answers'])

