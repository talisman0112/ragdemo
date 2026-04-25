import streamlit as st
import io
import sys
from contextlib import redirect_stdout
from knowledge_base import knowledge_bases_ervice

st.title("talisman")

# 初始化会话状态，避免每次重跑都丢失页面数据
if "file_info" not in st.session_state:
    st.session_state.file_info = None
if "file_text" not in st.session_state:
    st.session_state.file_text = ""
if "file_signature" not in st.session_state:
    st.session_state.file_signature = None
if "vec_result" not in st.session_state:
    st.session_state.vec_result = ""
if "vec_error" not in st.session_state:
    st.session_state.vec_error = None
if "services" not in st.session_state:
    # 将 service 放到 session_state 中，避免反复初始化
    st.session_state.services = {
        "knowledge_base": knowledge_bases_ervice(),
    }

uploaded_file = st.file_uploader(
    "请上传txt",
    type=["txt"],
    accept_multiple_files=False
)
services = st.session_state.services

if uploaded_file is not None:
    # 仅当上传了新文件时，才重新读取和解码
    current_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.file_signature != current_signature:
        file_name = uploaded_file.name
        file_type = uploaded_file.type
        file_size_bytes = uploaded_file.size

        content_bytes = uploaded_file.read()
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # 如果不是 utf-8，可以尝试 gbk（Windows 常见）
            text = content_bytes.decode("gbk", errors="replace")

        st.session_state.file_info = {
            "name": file_name,
            "type": file_type,
            "size_kb": file_size_bytes / 1024,
        }
        st.session_state.file_text = text
        st.session_state.file_signature = current_signature
        st.session_state.vec_result = ""
        st.session_state.vec_error = None

        # 调用向量化入库，并把输出结果保存到页面状态
        stdout_buffer = io.StringIO()
        error_msg = None
        
        with st.spinner("正在写入向量库，请稍候..."):
            try:
                with redirect_stdout(stdout_buffer):
                    services["knowledge_base"].change_to_vec(text, file_name)
            except Exception as e:
                error_msg = str(e)
                st.session_state.vec_error = error_msg
                
        result_text = stdout_buffer.getvalue().strip()
        if error_msg:
            st.session_state.vec_result = result_text + f"\n\n[错误] {error_msg}"
        elif result_text:
            st.session_state.vec_result = result_text
        else:
            st.session_state.vec_result = "change_to_vec 已执行（无打印输出）"

if st.session_state.file_info is not None:
    st.subheader(f"文件名：{st.session_state.file_info['name']}")
    st.write(f"格式：{st.session_state.file_info['type']}")
    st.write(f"大小：{st.session_state.file_info['size_kb']:.2f} KB")
    st.subheader("文本内容预览")

    # 只显示前 N 个字符，避免超大文本把页面撑爆
    max_chars = 20000
    text = st.session_state.file_text
    if len(text) > max_chars:
        st.text(text[:max_chars] + "\n\n...（已截断显示）")
    else:
        st.text(text)

    st.subheader("change_to_vec 输出")
    if st.session_state.vec_error:
        st.error(st.session_state.vec_result)
    else:
        st.text(st.session_state.vec_result)
