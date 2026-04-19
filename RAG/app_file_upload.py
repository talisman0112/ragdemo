import streamlit as st

st.title("talisman")

uploaded_file = st.file_uploader(
    "请上传txt",
    type=["txt"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size_bytes = uploaded_file.size
    file_size_kb = file_size_bytes / 1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type}")
    st.write(f"大小：{file_size_kb:.2f} KB")

    # 读取文本内容
    # streamlit 上传文件是 UploadedFile，通常可用 .read() 读取 bytes
    content_bytes = uploaded_file.read()

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 如果不是 utf-8，可以尝试 gbk（Windows 常见）
        text = content_bytes.decode("gbk", errors="replace")

    st.subheader("文本内容预览")

    # 可选：只显示前 N 个字符，避免超大文本把页面撑爆
    max_chars = 20000
    if len(text) > max_chars:
        st.text(text[:max_chars] + "\n\n...（已截断显示）")
    else:
        st.text(text)


