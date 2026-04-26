"""
Streamlit 问答界面：RAG 对话 + 本地历史会话（chat_history.json）。
运行（在 RAG 目录下）：streamlit run app_qa.py
或在仓库根目录：streamlit run RAG/app_qa.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

_RAG_DIR = Path(__file__).resolve().parent
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

import streamlit as st

from history import ChatHistoryStore
from rag import RAG, format_session_messages_for_prompt

st.set_page_config(page_title="Talisman 问答", page_icon="💬", layout="centered")

# 主区底部留白：避免最后几条消息被底部输入条遮挡（chat_input 由 Streamlit 固定在视口下方）
st.markdown(
    """
<style>
    section.main > div.block-container {
        padding-bottom: 5.5rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- session_state 初始化 ----------
if "history_store" not in st.session_state:
    st.session_state.history_store = ChatHistoryStore()

if "rag" not in st.session_state:
    with st.spinner("正在加载向量库与模型，首次可能较慢…"):
        st.session_state.rag = RAG()

if "session_id" not in st.session_state:
    # 证据驱动修复：此前会在状态丢失时无条件 create_session，导致出现空会话并“看似自动新建对话”
    existing = st.session_state.history_store.list_sessions(limit=1)
    if existing:
        st.session_state.session_id = existing[0]["id"]
    else:
        st.session_state.session_id = st.session_state.history_store.create_session()

if "_loaded_sid" not in st.session_state:
    st.session_state._loaded_sid = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

if "diag_logs" not in st.session_state:
    st.session_state.diag_logs = []


def _diag(event: str, **fields: object) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    info = " ".join([f"{k}={v}" for k, v in fields.items()])
    line = f"[{ts}] {event}" + (f" | {info}" if info else "")
    st.session_state.diag_logs.append(line)
    # 限制日志数量，避免会话过长
    st.session_state.diag_logs = st.session_state.diag_logs[-200:]


def _sync_messages_from_store() -> None:
    sid = st.session_state.session_id
    sess = st.session_state.history_store.get_session(sid)
    if not sess:
        st.session_state.messages = []
    else:
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in sess.get("messages") or []
        ]
    st.session_state._loaded_sid = sid


if st.session_state._loaded_sid != st.session_state.session_id:
    _diag("sync_messages", session_id=st.session_state.session_id)
    _sync_messages_from_store()

# ---------- 侧边栏：会话列表 ----------
with st.sidebar:
    st.header("会话")
    st.session_state.dev_mode = st.toggle("开发模式诊断", value=st.session_state.dev_mode)
    if st.session_state.dev_mode and st.button("清空诊断日志", use_container_width=True):
        st.session_state.diag_logs = []
        st.rerun()

    if st.button("新对话", use_container_width=True):
        st.session_state.session_id = st.session_state.history_store.create_session()
        _diag("new_session_clicked", new_session_id=st.session_state.session_id)
        st.session_state._loaded_sid = None
        st.session_state["history_session_select"] = st.session_state.session_id
        st.rerun()

    summaries = st.session_state.history_store.list_sessions()
    if not summaries:
        st.caption("暂无历史，发送一条消息后会自动保存。")
    else:
        ids = [s["id"] for s in summaries]
        _sb_key = "history_session_select"

        if st.session_state.session_id not in ids:
            _diag(
                "current_session_missing_in_ids",
                old_session_id=st.session_state.session_id,
                fallback_session_id=ids[0],
            )
            st.session_state.session_id = ids[0]
            st.session_state._loaded_sid = None

        # 仅当 key 缺失/失效时回填，避免覆盖用户刚做出的选择
        if _sb_key not in st.session_state or st.session_state[_sb_key] not in ids:
            st.session_state[_sb_key] = st.session_state.session_id

        def _fmt(sid: str) -> str:
            s = next(x for x in summaries if x["id"] == sid)
            t = (s.get("title") or "新对话")[:40]
            n = s.get("message_count", 0)
            return f"{t} · {n} 条"

        st.selectbox(
            "切换到",
            ids,
            format_func=_fmt,
            key=_sb_key,
        )

        if st.session_state[_sb_key] != st.session_state.session_id:
            _diag(
                "session_switch",
                from_session=st.session_state.session_id,
                to_session=st.session_state[_sb_key],
            )
            st.session_state.session_id = st.session_state[_sb_key]
            st.session_state._loaded_sid = None
            st.rerun()

        if st.button("删除当前会话", type="secondary", use_container_width=True):
            cur = st.session_state.session_id
            st.session_state.history_store.delete_session(cur)
            remaining = st.session_state.history_store.list_sessions(limit=1)
            if remaining:
                st.session_state.session_id = remaining[0]["id"]
            else:
                st.session_state.session_id = st.session_state.history_store.create_session()
            _diag("delete_session", deleted=cur, active_session=st.session_state.session_id)
            st.session_state._loaded_sid = None
            st.session_state["history_session_select"] = st.session_state.session_id
            st.rerun()

# ---------- 主区：聊天 ----------
st.title("Talisman 问答")
st.caption(
    "基于知识库的问答；多轮对话会写入当前会话；回复为流式输出。"
    " 输入框固定在窗口最下方（使用 Streamlit 自带底栏）。"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.dev_mode:
    with st.expander("诊断日志（开发模式）", expanded=True):
        st.caption(
            f"session_id={st.session_state.session_id} "
            f"loaded_sid={st.session_state._loaded_sid} "
            f"messages={len(st.session_state.messages)}"
        )
        st.code("\n".join(st.session_state.diag_logs[-80:]) or "(暂无日志)", language="text")

# st.chat_input 由框架固定在视口底部居中，不受父级 transform 影响（自写 CSS 固定 stForm 常失效）
if prompt := st.chat_input("输入问题，按 Enter 发送", key="qa_chat_input"):
    q = prompt.strip()
    if q:
        _diag("submit_prompt", session_id=st.session_state.session_id, q=q[:40])
        st.session_state.messages.append({"role": "user", "content": q})
        # 立即渲染用户输入，避免首次提交时看起来“无反应”
        with st.chat_message("user"):
            st.markdown(q)

        accumulated: dict[str, str] = {"text": ""}
        prior_turns = st.session_state.messages[:-1]
        short_term = format_session_messages_for_prompt(prior_turns)
        if st.session_state.dev_mode:
            _diag(
                "short_term_memory",
                prior_turns=len(prior_turns),
                memory_chars=len(short_term),
            )

        def _token_stream():
            try:
                for token in st.session_state.rag.stream_answer(
                    q, chat_history=short_term if short_term else None
                ):
                    accumulated["text"] += token
                    yield token
            except Exception as e:
                err = f"调用失败：{e}"
                accumulated["text"] += err
                yield err

        with st.chat_message("assistant"):
            st.write_stream(_token_stream)

        answer = accumulated["text"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.history_store.append_turn(
            st.session_state.session_id, q, answer
        )
        _diag(
            "append_turn",
            session_id=st.session_state.session_id,
            answer_chars=len(answer),
        )
        st.rerun()
