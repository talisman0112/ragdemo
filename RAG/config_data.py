import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dashscope_embeddings import DashScopeEmbeddings

load_dotenv(override=False)


# ==================== 路径配置 ====================
# 项目根目录、MD5记录文件、Chroma数据库持久化路径

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md5_path = os.path.join(PROJECT_ROOT, "md5.txt")
collection_name = "knowledge_base"
persist_directory = os.path.join(PROJECT_ROOT, "db")


# ==================== API 配置 ====================
# 阿里云百炼 DashScope API Key

api_key = os.environ.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError(
        "未设置 DASHSCOPE_API_KEY 环境变量。\n"
        "请在系统环境变量中配置（推荐）：\n"
        "  DASHSCOPE_API_KEY=sk-your-aliyun-key\n"
        "或在项目根目录的 .env 文件中设置。"
    )


# ==================== Embedding 配置 ====================
# 文本向量化模型：将文本转换为向量，用于语义检索

embedding_model = "text-embedding-v3"
embedding_function = DashScopeEmbeddings(api_key=api_key, model=embedding_model)
print(f"[Config] 使用阿里云百炼服务，模型: {embedding_model}")


# ==================== 文本分割器配置 ====================
# 将长文档切分成小块，便于向量化和检索

spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", "", "?", ".", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "`", "~", "-", "_", "+", "=", ":", ";", "\"", "'", "\""],
    length_function=len
)


# ==================== 数据库写入配置 ====================
# 控制向量数据库的批量写入和重试策略

start_chunk_size = 1000
write_batch_size = 20
write_max_retries = 3
write_retry_sleep_seconds = 1.5
# ==================== 检索相似文件 ====================
k = 1
# ==================== 聊天模型配置 ====================

chat_model = "qwen3-max"
chat_model_provider = "dashscope"
