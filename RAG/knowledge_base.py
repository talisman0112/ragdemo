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

    def change_to_vec(self,data:str,filename:str):
        md5_str = get_string_md5(data)
        if check_md5(md5_str):
            print(f"文件{filename}已存在，跳过")
            return
        else:
            if len(data) > config_data.start_chunk_size:
                chunks = self.spliter.split_text(data)
                for chunk in chunks:
                    self.chroma.add_texts([chunk],metadatas=[{"source":filename,"md5":md5_str,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
            else:
                self.chroma.add_texts([data],metadatas=[{"source":filename,"md5":md5_str,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
            save_md5(md5_str)
            print(f"文件{filename}已保存")