"""
诊断向量库写入问题
"""
import os
import sys

print("=" * 60)
print("诊断：向量库写入问题")
print("=" * 60)

# 1. 检查环境变量
print("\n[1] 检查环境变量...")
aliyun_key = os.environ.get("DASHSCOPE_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

if aliyun_key:
    print(f"  ✓ DASHSCOPE_API_KEY: {aliyun_key[:10]}...{aliyun_key[-4:]}")
else:
    print(f"  ✗ DASHSCOPE_API_KEY: 未设置")

if openai_key:
    print(f"  ✓ OPENAI_API_KEY: {openai_key[:10]}...{openai_key[-4:]}")
else:
    print(f"  ✗ OPENAI_API_KEY: 未设置")

# 2. 测试 API 连接
print("\n[2] 测试 Embedding API 连接...")
try:
    from openai import OpenAI
    
    if aliyun_key:
        client = OpenAI(
            api_key=aliyun_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print("  使用阿里云百炼服务")
    elif openai_key:
        client = OpenAI(api_key=openai_key)
        print("  使用 OpenAI 官方服务")
    else:
        print("  ✗ 没有可用的 API Key")
        sys.exit(1)
    
    # 测试生成 embedding
    test_text = "测试文本"
    print(f"  测试生成 embedding: '{test_text}'")
    
    response = client.embeddings.create(
        model="text-embedding-v3" if aliyun_key else "text-embedding-3-small",
        input=test_text
    )
    
    embedding_dim = len(response.data[0].embedding)
    print(f"  ✓ Embedding 生成成功，维度: {embedding_dim}")
    
except Exception as e:
    print(f"  ✗ Embedding 生成失败: {e}")

# 3. 测试 Chroma 向量库
print("\n[3] 测试 Chroma 向量库...")
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RAG"))
    from config_data import embedding_function, persist_directory, collection_name
    from langchain_chroma import Chroma
    
    print(f"  持久化目录: {persist_directory}")
    print(f"  集合名称: {collection_name}")
    
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    print("  ✓ Chroma 初始化成功")
    
    # 测试写入
    test_doc = "这是一个测试文档"
    print(f"  测试写入: '{test_doc}'")
    
    chroma.add_texts([test_doc], metadatas=[{"source": "test", "test": True}])
    print("  ✓ 写入成功")
    
    # 测试查询
    results = chroma.similarity_search(test_doc, k=1)
    print(f"  ✓ 查询成功，找到 {len(results)} 条结果")
    
except Exception as e:
    print(f"  ✗ Chroma 操作失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试知识库服务
print("\n[4] 测试知识库服务...")
try:
    from knowledge_base import knowledge_bases_ervice
    
    service = knowledge_bases_ervice()
    print("  ✓ 知识库服务初始化成功")
    
    test_content = "这是测试内容，用于验证写入功能。"
    print(f"  测试 change_to_vec: '{test_content[:20]}...'")
    
    service.change_to_vec(test_content, "test_file.txt")
    print("  ✓ change_to_vec 执行完成")
    
except Exception as e:
    print(f"  ✗ 知识库服务测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
