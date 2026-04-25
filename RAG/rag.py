from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi

from langchain_core.documents import Document  # 建议改掉你原来的 xml.dom.minidom.Document

from vectoc_store import VectocStore
import config_data as config


class RAG:
    def __init__(self):
        self.vectoc_store = VectocStore(config.embedding_function)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
你是talisman，一个智能助手，你的任务是根据以下上下文回答问题：
{context}
问题：{question}
答案：
""".strip()
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model,
            api_key=config.api_key,   # 你 config_data 里没有 api_key=QWEN_API_KEY，只有 DASHSCOPE 的 api_key
            temperature=0
        )

        self.chain = self.get_chain()

    def format_documents(self, documents: list[Document]):
        if not documents:
            return "没有找到相关内容"
        return "\n".join([doc.page_content for doc in documents])

    def get_chain(self):
        retriever = self.vectoc_store.get_retriever()

        chain = (
            {
                "question": RunnablePassthrough(),
                "context": retriever | self.format_documents,  # 每次都会用 question 检索
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        return chain


if __name__ == "__main__":
    rag = RAG()
    response = rag.chain.invoke("李倍乐是谁，他会干什么")
    print(response)