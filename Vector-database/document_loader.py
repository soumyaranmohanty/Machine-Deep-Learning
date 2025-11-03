"""
Document loader for various file formats
"""
import os
from typing import List, Dict, Any
from pathlib import Path
import logging

# Import specific document loaders
try:
    import PyPDF2
    from docx import Document as DocxDocument
    from bs4 import BeautifulSoup
except ImportError as e:
    logging.warning(f"Some document processing libraries not available: {e}")

#from config import RAGConfig

class DocumentLoader:
    """Load and preprocess documents from various file formats"""
    
    def __init__(self):
        self.supported_extensions = (".txt", ".pdf", ".docx", ".md")
        self.logger = logging.getLogger(__name__)
        
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single document and return its content with metadata
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check if file extension is supported
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        # Load content based on file type
        content = ""
        if file_path.suffix.lower() == ".txt":
            content = self._load_txt(file_path)
        elif file_path.suffix.lower() == ".pdf":
            content = self._load_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            content = self._load_docx(file_path)
        elif file_path.suffix.lower() == ".md":
            content = self._load_txt(file_path)  # Markdown can be loaded as text
        else:
            raise ValueError(f"Handler not implemented for: {file_path.suffix}")
            
        # Create document metadata
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower(),
            "content_length": len(content)
        }
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of document dictionaries
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        documents = []
        
        # Walk through directory and subdirectories
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                    self.logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
                    
        return documents
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file content"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                    
                return content.strip()
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file content"""
        try:
            doc = DocxDocument(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
                
            return content.strip()
        except Exception as e:
            self.logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def _load_html(self, file_path: Path) -> str:
        """Load HTML file content (extract text)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            self.logger.error(f"Error reading HTML {file_path}: {e}")
            return ""
