# Use a pipeline as a high-level helper
from typing import Iterable

from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

EMBEDDINGS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def get_db(path: str, embeddings: Embeddings) -> Chroma:
    return Chroma(persist_directory=path, embedding_function=embeddings)


def get_embeddings(embeddings_model: str = EMBEDDINGS_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=embeddings_model)


def get_similar_documents(query: str, db: Chroma):
    return db.similarity_search(query)


def get_similar_chunks(documents: list[Document]) -> Iterable[str]:
    for document in documents:
        yield document.page_content


if __name__ == "__main__":
    db = get_db("./emb", get_embeddings())
    while True:
        query = input("Query :\n")
        docs = get_similar_documents(query, db)
        for doc in docs:
            print("> ", doc.page_content)
