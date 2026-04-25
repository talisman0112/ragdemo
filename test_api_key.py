"""
检测 API Key 是否有效
支持阿里云百炼和 OpenAI 官方
"""
import os
import sys

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(override=False)

# 获取 API Key（优先阿里云百炼）
ALIYUN_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

api_key = ALIYUN_KEY or OPENAI_KEY
use_aliyun = ALIYUN_KEY is not None

if not api_key:
    print("❌ 未找到 API Key 环境变量")
    print("请在 .env 文件或系统环境变量中设置：")
    print("  DASHSCOPE_API_KEY=sk-your-aliyun-key  （阿里云百炼）")
    print("  OPENAI_API_KEY=sk-your-openai-key      （OpenAI 官方）")
    sys.exit(1)

print(f"使用服务: {'阿里云百炼' if use_aliyun else 'OpenAI 官方'}")
print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
print("正在测试连接...\n")

try:
    from openai import OpenAI
    
    if use_aliyun:
        # 阿里云百炼兼容接口
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        # OpenAI 官方
        client = OpenAI(api_key=api_key)
    
    # 测试：列出模型（不消耗额度）
    models = client.models.list()
    print("✅ API Key 有效！")
    print(f"可用模型数: {len(models.data)}")
    
    # 显示 Embedding 相关模型
    embedding_models = [m.id for m in models.data if 'embedding' in m.id.lower()]
    if embedding_models:
        print(f"Embedding 模型: {embedding_models[:3]}")
    
except Exception as e:
    print(f"❌ API Key 无效或连接失败")
    print(f"错误: {e}")
    sys.exit(1)
