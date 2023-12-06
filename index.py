from llama_index import (
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
from dotenv import load_dotenv
from glob import glob

PERSIST_DIR = "index"
MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1024


def get_files(regex: str):
    return glob(regex)


def get_service_context(model: str = MODEL, chunk_size: int = CHUNK_SIZE):
    chatgpt = OpenAI(temperature=0, model=model)
    return ServiceContext.from_defaults(llm=chatgpt, chunk_size=chunk_size)


def get_document_index(_docs: list, service_context):
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True
    )
    return DocumentSummaryIndex.from_documents(
        _docs,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )


def load_query_engine(query_engine: str = PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=query_engine)
    doc_summary_index = load_index_from_storage(storage_context)
    return doc_summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True
    )


if __name__ == "__main__":
    load_dotenv()

    docs = SimpleDirectoryReader(input_files=get_files("data-test/*.pdf")).load_data()

    docs_index = get_document_index(docs, get_service_context())
    docs_index.storage_context.persist(PERSIST_DIR)
