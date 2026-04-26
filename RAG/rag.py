from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi

from langchain_core.documents import Document  # 建议改掉你原来的 xml.dom.minidom.Document

from vectoc_store import VectocStore
import config_data as config

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from history import ChatHistoryStore


def format_session_messages_for_prompt(messages: list[dict[str, Any]], *, max_chars: int = 12000) -> str:
    """将当前会话消息列表格式化为提示词中的短期记忆文本（过长则截断尾部）。"""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            prefix = "用户"
        elif role == "assistant":
            prefix = "助手"
        else:
            prefix = str(role)
        lines.append(f"{prefix}：{content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "…（更早对话已省略）\n" + text[-max_chars:]
    return text


class RAG:
    def __init__(self):
        self.vectoc_store = VectocStore(config.embedding_function)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""
你是talisman，一个智能助手，你的任务是根据以下上下文回答问题：
{context}

以下为本轮会话的短期记忆（仅用于理解指代与承接上文，事实仍以知识库上下文为准）：
{chat_history}

当前用户问题：{question}
答案：
""".strip()
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model,
            api_key=config.api_key,   # 你 config_data 里没有 api_key=QWEN_API_KEY，只有 DASHSCOPE 的 api_key
            temperature=0
        )

        self.llm_chain = self.prompt_template | self.chat_model | StrOutputParser()
        self.chain = self.get_chain()

    def format_documents(self, documents: list[Document]):
        if not documents:
            return "没有找到相关内容"
        return "\n".join([doc.page_content for doc in documents])

    def _rag_inputs(self, payload: dict[str, Any]) -> dict[str, str]:
        """仅用 question 做向量检索；chat_history 仅进入提示词。"""
        question = payload["question"]
        raw_hist = payload.get("chat_history")
        if isinstance(raw_hist, str) and raw_hist.strip():
            chat_history = raw_hist.strip()
        else:
            chat_history = "（本轮会话尚无更早消息）"
        retriever = self.vectoc_store.get_retriever()
        docs = retriever.invoke(question)
        context = self.format_documents(docs)
        return {"context": context, "chat_history": chat_history, "question": question}

    def get_chain(self):
        return RunnableLambda(self._rag_inputs) | self.llm_chain

    def stream_answer(
        self,
        question: str,
        *,
        chat_history: str | None = None,
    ) -> Iterator[str]:
        """先同步检索，再对 LLM 输出做流式分片（与 invoke 同源上下文）。"""
        llm_in = self._rag_inputs({"question": question, "chat_history": chat_history})
        for piece in self.llm_chain.stream(llm_in):
            if piece is None:
                continue
            if isinstance(piece, str):
                yield piece
            else:
                raw = getattr(piece, "content", None)
                if isinstance(raw, str) and raw:
                    yield raw
                elif isinstance(raw, list):
                    for part in raw:
                        if isinstance(part, str) and part:
                            yield part
                        elif hasattr(part, "text") and part.text:
                            yield str(part.text)

    def invoke(
        self,
        question: str,
        *,
        chat_history: str | None = None,
        history_store: Optional["ChatHistoryStore"] = None,
        session_id: Optional[str] = None,
    ) -> str:
        answer = self.chain.invoke({"question": question, "chat_history": chat_history})
        if history_store is not None and session_id:
            history_store.append_turn(session_id, question, answer)
        return answer

