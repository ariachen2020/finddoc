from PyPDF2 import PdfReader
from docx import Document
import jieba
from typing import Tuple, List

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 100

    def process_file(self, file) -> Tuple[List[str], List[int]]:
        """處理上傳的文件，返回文本塊和對應的頁碼"""
        file_name = file.name.lower()
        
        if file_name.endswith('.pdf'):
            return self._process_pdf(file)
        elif file_name.endswith('.docx'):
            return self._process_docx(file)
        elif file_name.endswith('.txt'):
            return self._process_txt(file)
        else:
            raise ValueError("不支援的文件格式")

    def _process_pdf(self, file) -> Tuple[List[str], List[int]]:
        """處理 PDF 文件"""
        reader = PdfReader(file)
        text_chunks = []
        page_numbers = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text.strip():
                continue
                
            chunks = self._split_text(text)
            text_chunks.extend(chunks)
            page_numbers.extend([page_num] * len(chunks))
            
        return text_chunks, page_numbers

    def _process_docx(self, file) -> Tuple[List[str], List[int]]:
        """處理 Word 文件"""
        doc = Document(file)
        text_chunks = []
        page_numbers = []
        current_page = 1
        
        full_text = ""
        for para in doc.paragraphs:
            full_text += para.text + "\n"
            
        chunks = self._split_text(full_text)
        text_chunks.extend(chunks)
        page_numbers.extend([current_page] * len(chunks))
        
        return text_chunks, page_numbers

    def _process_txt(self, file) -> Tuple[List[str], List[int]]:
        """處理純文本文件"""
        text = file.read().decode('utf-8')
        chunks = self._split_text(text)
        page_numbers = [1] * len(chunks)  # 純文本文件視為單頁
        
        return chunks, page_numbers

    def _split_text(self, text: str) -> List[str]:
        """將文本分割成小塊"""
        words = list(jieba.cut(text))
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word)
            
            if current_length >= self.chunk_size:
                chunks.append("".join(current_chunk))
                # 保留一部分文本作為重疊
                overlap_words = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_words
                current_length = sum(len(w) for w in overlap_words)
        
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        return chunks 