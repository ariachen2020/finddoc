from typing import List, Tuple
import jieba
import logging
import warnings
from difflib import SequenceMatcher

# 禁用 jieba 的警告
warnings.filterwarnings("ignore", category=SyntaxWarning)
# 禁用 jieba 的日誌輸出
jieba.setLogLevel(logging.INFO)

class ChromaDBManager:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        
    def string_similarity(self, a: str, b: str) -> float:
        """計算兩個字符串的相似度"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def store_documents(self, documents: List[str], page_numbers: List[int]):
        """存儲文檔"""
        if not documents:
            return
            
        # 清除舊數據
        self.documents = documents
        self.metadatas = [{"page": str(page)} for page in page_numbers]

    def search_similar(self, query: str, top_k: int = 10) -> Tuple[List[str], List[dict]]:
        """搜索相似文檔"""
        if not self.documents:
            return [], []

        # 計算每個文檔片段與查詢的相似度
        similarities = []
        query_words = set(jieba.cut(query))
        
        for doc in self.documents:
            # 1. 關鍵詞匹配
            doc_words = set(jieba.cut(doc))
            word_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            
            # 2. 字符串相似度
            string_sim = self.string_similarity(query, doc)
            
            # 3. 計算總分數 (加權平均)
            score = (word_overlap * 0.7) + (string_sim * 0.3)
            similarities.append(score)
        
        # 獲取最相似的文檔
        sorted_indices = sorted(range(len(similarities)), 
                              key=lambda i: similarities[i], 
                              reverse=True)[:top_k]
        
        result_docs = [self.documents[i] for i in sorted_indices if similarities[i] > 0]
        result_metadata = [self.metadatas[i] for i in sorted_indices if similarities[i] > 0]
        
        return result_docs, result_metadata 