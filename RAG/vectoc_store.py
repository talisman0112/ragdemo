import config_data as config
from langchain_chroma import Chroma

class VectocStore:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.vectoc_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=embedding_function,
            persist_directory=config.persist_directory,
        )
    def get_retriever(self):
        return self.vectoc_store.as_retriever(
            search_kwargs={"k": getattr(config, "k", config.k)}
        )

if __name__ == "__main__":
    store = VectocStore(config.embedding_function)
    # 1) 先打印当前使用的路径/collection
    print("persist_directory =", config.persist_directory)
    print("collection_name =", config.collection_name)
    docs = store.get_retriever().invoke("李倍乐是谁")
    print("retrieval docs len =", len(docs))
    print(docs[:3])