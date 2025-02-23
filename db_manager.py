import chromadb
from chromadb.config import Settings
import os
import tempfile
from jieba import lcut  # 中文分詞
from gensim.models import KeyedVectors
import numpy as np
from difflib import SequenceMatcher
import jieba
import logging
from typing import List, Tuple
import tiktoken  # 用於計算 token 數量

# 禁用 jieba 的日誌輸出
jieba.setLogLevel(logging.INFO)

class ChromaDBManager:
    def __init__(self):
        # 使用內存模式
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=":memory:",  # 使用內存模式
                anonymized_telemetry=False
            )
        )
        
        try:
            self.collection = self.client.get_or_create_collection(name="document_qa")
        except Exception as e:
            print(f"Error initializing collection: {e}")
            self.collection = None
        
        self.chunk_size = 500
        self.chunk_overlap = 100
        
        # 同義詞字典
        self.synonyms = {
            "價格": ["金額", "費用", "成本", "價位"],
            "時間": ["時候", "日期", "期間", "時段"],
            "位置": ["地點", "地方", "場所", "處所"],
        }

    def store_documents(self, documents, page_numbers):
        """存儲文檔到向量數據庫"""
        if not documents:
            return
            
        if self.collection is None:
            try:
                self.collection = self.client.get_or_create_collection(name="document_qa")
            except Exception as e:
                print(f"Error creating collection: {e}")
                return

        try:
            # 準備數據
            ids = [f"doc_{i}" for i in range(len(documents))]
            metadatas = [{"page": str(page)} for page in page_numbers]
            
            # 添加文檔
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"Error storing documents: {e}")

    def string_similarity(self, a, b):
        """計算兩個字符串的相似度"""
        return SequenceMatcher(None, a, b).ratio()

    def get_synonyms(self, word):
        """獲取詞的同義詞"""
        for key, values in self.synonyms.items():
            if word in [key] + values:
                return [key] + values
        return [word]

    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        return len(self.tokenizer.encode(text))

    def search_similar(self, query: str, top_k: int = 10) -> Tuple[List[str], List[dict]]:
        """搜索相似文檔"""
        if self.collection is None:
            return [], []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results['documents']:
                return [], []
                
            return results['documents'][0], results['metadatas'][0]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return [], []

    def process_text(self, text):
        """文本分塊處理"""
        chunks = []
        start = 0
        
        while start < len(text):
            # 找到一個合適的切分點
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # 尋找最近的句號或換行符
            while end < len(text) and end - start < self.chunk_size + 50:
                if text[end] in ['。', '！', '？', '\n']:
                    end += 1
                    break
                end += 1
                
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 考慮重疊
            start = end - self.chunk_overlap
            
        return chunks 