from flatbuffers.flexbuffers import Object
import config_data
import hashlib
import json
import os
import re
import time
import uuid
import warnings
import requests
import urllib.parse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

def save_md5(md5_str: str) -> None:
    with open(config_data.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def check_md5(md5_str: str) -> bool:
    """是否与已记录的全文 MD5 重复（精确去重，与向量相似度无关）。"""
    if not os.path.exists(config_data.md5_path):
        open(config_data.md5_path, "w", encoding="utf-8").close()
        return False
    with open(config_data.md5_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            existing = line.split()[0]
            if existing == md5_str:
                return True
    return False

def get_string_md5(input_str:str,encoding='utf-8'):
    str_bytes = input_str.encode(encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()
    
    
class knowledge_bases_ervice(Object):
    def __init__(self):
        os.makedirs(config_data.persist_directory,exist_ok=True)
        self.chroma= Chroma(collection_name=config_data.collection_name,embedding_function=config_data.embedding_function,persist_directory=config_data.persist_directory)
        self.spliter=config_data.spliter

    def _add_texts_with_retry(self, texts, metadatas, desc=""):
        """带重试机制的写入方法"""
        for retry in range(config_data.write_max_retries):
            try:
                self.chroma.add_texts(texts, metadatas=metadatas)
                return True
            except Exception as e:
                if retry == config_data.write_max_retries - 1:
                    raise RuntimeError(f"{desc}写入失败（已重试{config_data.write_max_retries}次）: {e}") from e
                print(f"{desc}写入超时，第{retry + 1}次重试...")
                time.sleep(config_data.write_retry_sleep_seconds)
        return False

    def change_to_vec(self,data:str,filename:str):
        start_time = time.time()
        md5_str = get_string_md5(data)
        if check_md5(md5_str):
            print(f"文件{filename}已存在，跳过")
            return
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if len(data) > config_data.start_chunk_size:
                chunks = self.spliter.split_text(data)
                batch_size = max(1, config_data.write_batch_size)
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(chunks))
                    batch_chunks = chunks[start:end]
                    batch_metadatas = [
                        {"source": filename, "md5": md5_str, "timestamp": timestamp}
                        for _ in batch_chunks
                    ]
                    
                    if self._add_texts_with_retry(batch_chunks, batch_metadatas, f"第{batch_idx + 1}/{total_batches}批"):
                        print(f"已完成批次 {batch_idx + 1}/{total_batches}")
                        
                chunk_count = len(chunks)
            else:
                # 小文件也使用重试机制
                if self._add_texts_with_retry(
                    [data],
                    [{"source": filename, "md5": md5_str, "timestamp": timestamp}],
                    "小文件"
                ):
                    print(f"小文件已写入")
                chunk_count = 1
                
            save_md5(md5_str)
            elapsed = time.time() - start_time
            print(f"文件{filename}已保存，chunk数：{chunk_count}，耗时：{elapsed:.2f}s")
