import chromadb
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
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection("documents")
            self.chunk_size = 500  # 調整分塊大小
            self.chunk_overlap = 100  # 增加重疊以避免切斷重要信息
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            
            # 同義詞字典（可以根據需要擴充）
            self.synonyms = {
                "價格": ["金額", "費用", "成本", "價位"],
                "時間": ["時候", "日期", "期間", "時段"],
                "位置": ["地點", "地方", "場所", "處所"],
                # 可以繼續添加更多同義詞
            }
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.client = None

    def store_documents(self, text_chunks, page_numbers):
        """儲存文件片段和對應的頁碼"""
        try:
            # 生成唯一ID
            ids = [f"doc_{i}" for i in range(len(text_chunks))]
            
            # 將頁碼資訊加入 metadata
            metadatas = [{"page": str(page)} for page in page_numbers]
            
            # 儲存到資料庫
            self.collection.add(
                documents=text_chunks,
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

    def search_similar(self, query: str, top_k: int = 10, max_tokens: int = 6000) -> Tuple[List[str], List[dict]]:
        query_words = lcut(query)
        expanded_words = []
        for word in query_words:
            expanded_words.extend(self.get_synonyms(word))
        expanded_words = list(set(expanded_words))

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # 計算相關性分數並限制 token 數量
        scored_results = []
        total_tokens = 0
        
        for doc, meta in zip(documents, metadatas):
            score = 0
            doc_words = lcut(doc)
            doc_tokens = self.count_tokens(doc)
            
            # 計算分數
            for word in expanded_words:
                if word in doc:
                    score += 1
            
            for i in range(len(query_words) - 1):
                query_phrase = query_words[i] + query_words[i + 1]
                for j in range(len(doc_words) - 1):
                    doc_phrase = doc_words[j] + doc_words[j + 1]
                    if self.string_similarity(query_phrase, doc_phrase) > 0.8:
                        score += 0.5
            
            scored_results.append((doc, meta, score, doc_tokens))
        
        # 根據分數排序
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        # 選擇最相關的文檔，同時確保不超過 token 限制
        filtered_docs = []
        filtered_metadata = []
        current_tokens = 0
        
        for doc, meta, _, doc_tokens in scored_results:
            if current_tokens + doc_tokens <= max_tokens:
                filtered_docs.append(doc)
                filtered_metadata.append(meta)
                current_tokens += doc_tokens
            else:
                break
        
        return filtered_docs, filtered_metadata

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