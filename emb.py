from langchain_community.embeddings import LlamafileEmbeddings

embeddings = LlamafileEmbeddings(base_url="http://localhost:8080")



doc_embeddings = embeddings.embed_documents(
    [
        "Alpha is the first letter of the Greek alphabet",
        "Beta is the second letter of the Greek alphabet",
    ]
)
query_embedding = embeddings.embed_query(
    "What is the second letter of the Greek alphabet"
)
print(doc_embeddings)