# SQLite 版本修復
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 其他導入
import chromadb
from chromadb.config import Settings
from typing import List, Tuple
import jieba
import logging

# 禁用 jieba 的日誌輸出
jieba.setLogLevel(logging.INFO)

class ChromaDBManager:
    def __init__(self):
        # 使用最簡單的內存設置
        self.client = chromadb.Client(
            Settings(
                is_persistent=False,  # 禁用持久化
                anonymized_telemetry=False
            )
        )
        
        try:
            self.collection = self.client.create_collection(name="document_qa")
        except Exception as e:
            print(f"Error initializing collection: {e}")
            self.collection = None

    def store_documents(self, documents, page_numbers):
        """存儲文檔到向量數據庫"""
        if not documents:
            return
            
        try:
            # 重新創建集合（清除舊數據）
            if self.collection is not None:
                self.client.delete_collection("document_qa")
            self.collection = self.client.create_collection(name="document_qa")
            
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

    def search_similar(self, query: str, top_k: int = 10) -> Tuple[List[str], List[dict]]:
        """搜索相似文檔"""
        if self.collection is None:
            return [], []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.collection.get()['ids']))
            )
            
            if not results['documents']:
                return [], []
                
            return results['documents'][0], results['metadatas'][0]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return [], [] 