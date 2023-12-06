from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    storage_context = StorageContext.from_defaults(persist_dir="index")
    doc_summary_index = load_index_from_storage(storage_context)
    query_engine = doc_summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True
    )

    while True:
        query = input("Question ?\n")

        response = query_engine.query(query)
        print(response)
