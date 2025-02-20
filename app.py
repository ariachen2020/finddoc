import streamlit as st
from document_processor import DocumentProcessor
from qa_engine import QAEngine
from db_manager import ChromaDBManager
from openai import OpenAI

def init_openai_api():
    """初始化 OpenAI API"""
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = ''
    
    api_key = st.text_input(
        "請輸入 OpenAI API Key",
        value=st.session_state.OPENAI_API_KEY,
        type="password"
    )
    
    if api_key:
        st.session_state.OPENAI_API_KEY = api_key
        client = OpenAI(api_key=api_key)
        return client
    return None

def main():
    st.title("DocQA - 文件快速問答助手")
    
    # API Key 設置
    client = init_openai_api()
    if client is None:
        st.warning("請先輸入 OpenAI API Key")
        return
    
    # 文件上傳區
    uploaded_file = st.file_uploader("上傳文件", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        # 處理文件
        doc_processor = DocumentProcessor()
        text_chunks, page_numbers = doc_processor.process_document(uploaded_file)
        
        # 儲存到向量資料庫
        db_manager = ChromaDBManager()
        if db_manager.client is None:
            st.error("無法初始化資料庫連接")
            return
        db_manager.store_documents(text_chunks, page_numbers)
        
        # 問答區域
        user_question = st.text_input("請輸入您的問題")
        
        if user_question:
            try:
                qa_engine = QAEngine(client)
                answer, source_pages = qa_engine.get_answer(user_question, db_manager)
                st.write("回答:", answer)
                st.write("來源頁碼:", source_pages)
            except Exception as e:
                st.error(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 