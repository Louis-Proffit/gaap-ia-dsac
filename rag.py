from typing import Iterable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model import get_embeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.document_loaders import PyPDFLoader
import glob

CHUNK_SIZE = 10000
CHUNK_OVERLAP_SIZE = 2000


def generate_paths(regex: str) -> list[str]:
    return glob.glob(regex)


def path_to_documents(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def documents_to_chunks(documents: list[Document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP_SIZE) -> Iterable[
    Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def get_db(chunks: list[Document], embeddings: Embeddings, output_path: str) -> Chroma:
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=output_path)


def main(files_regex: str, output_path: str):
    chunks = []
    for path in generate_paths(files_regex):
        print("Handling ", path)
        chunks += documents_to_chunks(path_to_documents(path))
    embeddings = get_embeddings()
    db = get_db(chunks=chunks, embeddings=embeddings, output_path=output_path)
    db.persist()


if __name__ == "__main__":
    main("data/*.pdf", "./embeddings")
