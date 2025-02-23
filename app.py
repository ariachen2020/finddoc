import streamlit as st
from document_processor import DocumentProcessor
from qa_engine import QAEngine
from db_manager import ChromaDBManager
from openai import OpenAI

def init_openai_api():
    """初始化 OpenAI API"""
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.error("請輸入 OpenAI API Key")
        return None
    return OpenAI(api_key=api_key)

def main():
    st.title("文件問答助手")
    
    # 初始化 OpenAI API
    client = init_openai_api()
    if client is None:
        return
        
    # 上傳文件
    uploaded_file = st.file_uploader("上傳文件", type=["txt", "pdf", "docx"])
    
    if uploaded_file:
        try:
            # 處理文件
            doc_processor = DocumentProcessor()
            text_chunks, page_numbers = doc_processor.process_file(uploaded_file)
            
            if not text_chunks:
                st.error("無法處理文件")
                return
                
            # 初始化資料庫管理器並儲存文件
            db_manager = ChromaDBManager()
            db_manager.store_documents(text_chunks, page_numbers)
            
            # 問答部分
            question = st.text_input("請輸入您的問題")
            
            if question:
                qa_engine = QAEngine(client)
                answer, sources = qa_engine.get_answer(question, db_manager)
                
                st.write("回答:", answer)
                if sources:
                    st.write("來源頁碼:", ", ".join(str(s.get("page", "未知")) for s in sources))
                    
        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 