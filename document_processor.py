import PyPDF2
from docx import Document
import tiktoken
from typing import List

class DocumentProcessor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_document(self, file):
        """處理上傳的文件，返回文本片段和對應頁碼"""
        text_chunks = []
        page_numbers = []
        
        if file.name.endswith('.pdf'):
            text_chunks, page_numbers = self._process_pdf(file)
        elif file.name.endswith('.docx'):
            text_chunks, page_numbers = self._process_docx(file)
        elif file.name.endswith('.txt'):
            text_chunks, page_numbers = self._process_txt(file)
            
        return text_chunks, page_numbers

    def _process_pdf(self, file):
        import io
        
        text_chunks = []
        page_numbers = []
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # 將頁面文本分割成較小的片段
            chunks = self._split_text(text)
            text_chunks.extend(chunks)
            # 為每個文本片段記錄對應的頁碼
            page_numbers.extend([page_num + 1] * len(chunks))
            
        return text_chunks, page_numbers

    def _process_docx(self, file):
        import io
        
        text_chunks = []
        page_numbers = []
        doc = Document(io.BytesIO(file.read()))
        
        # 注意：docx 不直接支持頁碼，這裡用段落序號代替
        for para_num, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                chunks = self._split_text(paragraph.text)
                text_chunks.extend(chunks)
                # 使用段落序號作為標識
                page_numbers.extend([para_num + 1] * len(chunks))
                
        return text_chunks, page_numbers

    def _process_txt(self, file):
        text_chunks = []
        page_numbers = []
        content = file.read().decode('utf-8')
        
        # 對於txt文件，以行數作為標識
        lines = content.split('\n')
        current_chunk = []
        current_line = 1
        
        for line in lines:
            if line.strip():
                current_chunk.append(line)
                if len(current_chunk) >= 3:  # 每3行作為一個chunk
                    text_chunks.append('\n'.join(current_chunk))
                    page_numbers.append(current_line)
                    current_chunk = []
            current_line += 1
            
        if current_chunk:  # 處理最後剩餘的行
            text_chunks.append('\n'.join(current_chunk))
            page_numbers.append(current_line)
            
        return text_chunks, page_numbers

    def _split_text(self, text, chunk_size=1000):
        """將文本分割成較小的片段"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in text.split('.'):
            sentence = sentence.strip() + '.'
            if current_size + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks 