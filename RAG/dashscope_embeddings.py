"""
阿里云百炼 DashScope Embedding 适配器
兼容 LangChain 接口，使用 HTTP API 直接调用
"""
import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings

class DashScopeEmbeddings(Embeddings):
    """阿里云百炼 DashScope Embedding 类"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-v3"):
        """
        初始化 DashScope Embeddings
        
        Args:
            api_key: DashScope API Key，默认从环境变量 DASHSCOPE_API_KEY 读取
            model: 模型名称，可选 text-embedding-v3, text-embedding-v2, text-embedding-v1
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 api_key 或设置 DASHSCOPE_API_KEY 环境变量")
        
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用 DashScope API 生成 embeddings
        """
        if not texts:
            return []
        
        # 确保所有文本都是字符串
        texts = [str(t) if t is not None else "" for t in texts]
        
        payload = {
            "model": self.model,
            "input": {
                "texts": texts
            }
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "output" not in data or "embeddings" not in data["output"]:
                raise RuntimeError(f"DashScope API 返回格式错误: {data}")
            
            embeddings = [item["embedding"] for item in data["output"]["embeddings"]]
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DashScope API 请求失败: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"DashScope API 响应解析失败: {e}, 响应: {data}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成文档 embedding 列表
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding 向量列表
        """
        return self._call_api(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成单个查询文本的 embedding
        
        Args:
            text: 查询文本
            
        Returns:
            embedding 向量
        """
        results = self._call_api([text])
        return results[0] if results else []


# 测试代码
if __name__ == "__main__":
    import os
    
    # 测试 embedding
    embeddings = DashScopeEmbeddings()
    
    # 单文本测试
    query = "测试文本"
    result = embeddings.embed_query(query)
    print(f"单文本 embedding 维度: {len(result)}")
    
    # 批量测试
    docs = ["第一段文本", "第二段文本", "第三段文本"]
    results = embeddings.embed_documents(docs)
    print(f"批量 embedding: {len(results)} 个文档，每维 {len(results[0])}")
