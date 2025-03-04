import openai
from typing import List
import os
from openai import OpenAI
import tiktoken

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QAEngine:
    def __init__(self, client):
        self.client = client
        self.model = "gpt-4"  # 使用穩定版 GPT-4 模型
        # 增加搜索參數設置
        self.search_top_k = 10  # 增加檢索數量
        self.max_tokens = 6000  # 為系統提示詞和用戶問題保留空間
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        return len(self.tokenizer.encode(text))

    def get_answer(self, question, db_manager):
        """根據問題生成答案，並返回來源頁碼"""
        # 系統提示詞的 tokens
        system_prompt = """你是一個專業的文件問答助手。
        1. 請仔細分析所有提供的段落
        2. 綜合所有相關信息提供完整答案
        3. 引用具體的段落編號說明信息來源
        4. 如果不同段落有互補信息，請整合在一起
        5. 對於模糊匹配的內容，請說明相關度和可信度"""
        
        system_tokens = self.count_tokens(system_prompt)
        question_tokens = self.count_tokens(question)
        
        # 計算剩餘可用的 tokens
        available_tokens = 8192 - system_tokens - question_tokens - 100  # 保留一些餘量
        
        # 搜尋相關文件片段
        relevant_docs, source_pages = db_manager.search_similar(
            question, 
            top_k=self.search_top_k
        )
        
        if not relevant_docs:
            return "抱歉，找不到相關資訊。", []
        
        # 組織上下文，同時控制 token 數量
        context_parts = []
        current_tokens = 0
        
        for i, doc in enumerate(relevant_docs, 1):
            part = f"段落{i}：{doc}"
            part_tokens = self.count_tokens(part)
            
            if current_tokens + part_tokens > available_tokens:
                break
                
            context_parts.append(part)
            current_tokens += part_tokens
        
        context = "\n\n".join(context_parts)
        
        # 使用 OpenAI API 生成答案
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"根據以下內容回答問題：\n\n{context}\n\n問題：{question}"}
                ],
                temperature=0.4  # 適度提高創造性
            )
            answer = response.choices[0].message.content
            
            # 只返回實際使用的段落對應的頁碼
            used_source_pages = source_pages[:len(context_parts)]
            
            return answer, used_source_pages
        except Exception as e:
            return f"生成答案時發生錯誤: {str(e)}", [] 