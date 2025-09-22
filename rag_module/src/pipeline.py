from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

class RAGPipeline:
    def __init__(
        self,
        data_dir: str = "../data",
        persist_dir: str = "chroma_index",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1500,
        chunk_overlap: int = 250
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            data_dir: Directory containing the documents
            persist_dir: Directory to persist the vector store
            model_name: Name of the embedding model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        load_dotenv()
        
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self._init_embeddings()
        self._init_llm()
        
    def _init_embeddings(self):
        """Initialize the embedding model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'}
        )
    
    def _init_llm(self):
        """Initialize the LLM"""
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat-v3.1:free",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.7,
            max_tokens=2000
        )
    
    def load_documents(self, file_pattern: str = "**/*.xlsx") -> List[Document]:
        """
        Load documents from the data directory.
        
        Args:
            file_pattern: Glob pattern for files to load
        
        Returns:
            List of loaded documents
        """
        loader = DirectoryLoader(
            self.data_dir,
            glob=file_pattern,
            loader_cls=UnstructuredExcelLoader
        )
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
        
        Returns:
            List of document chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(documents)
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create and persist the vector store.
        
        Args:
            documents: List of documents to index
        
        Returns:
            Chroma vector store instance
        """
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        vectorstore.persist()
        return vectorstore
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        Load an existing vector store if it exists.
        
        Returns:
            Chroma vector store instance or None if it doesn't exist
        """
        if not os.path.exists(self.persist_dir):
            return None
            
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
    
    def query_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Query the vector store.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        
        Returns:
            List of relevant documents
        """
        vectorstore = self.load_existing_vectorstore()
        if not vectorstore:
            raise ValueError("No vector store found. Please index documents first.")
        
        return vectorstore.similarity_search(query, k=k)
    
    def generate_response(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Generate a response using the LLM with visualization data.
        
        Args:
            query: User query
            context_docs: Relevant documents for context
        
        Returns:
            Dictionary containing response text and visualization data
        """
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = f"""Given the following context from documents, please provide both a text response and visualization data.

        Context:
        {context}
        
        Question: {query}
        
        Please provide your response in JSON format with these keys:
        1. "text_response": A clear and concise answer based on the context
        2. "visualization": JSON object with visualization data including:
           - "type": Chart type (bar, line, pie, etc.)
           - "title": Chart title
           - "data": Data points
           - "labels": Axis labels or categories
           - "colors": Color scheme
           Only include visualization data if it's relevant to represent the information graphically.
        3. "confidence_score": A float between 0 and 1 indicating confidence in the answer
        4. "sources": List of relevant source documents used

        Example format:
        {{
            "text_response": "Sales increased by 25% in Q4 2024",
            "visualization": {{
                "type": "line",
                "title": "Quarterly Sales Growth",
                "data": [10, 15, 20, 25],
                "labels": ["Q1", "Q2", "Q3", "Q4"],
                "colors": ["#1f77b4"]
            }},
            "confidence_score": 0.92,
            "sources": ["Q4 2024 Report", "Annual Review"]
        }}
        
        If visualization is not relevant, set it to null."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            # Parse the response as JSON
            import json
            parsed_response = json.loads(response.content)
            return parsed_response
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            return {
                "text_response": response.content,
                "visualization": None,
                "confidence_score": 0.5,
                "sources": [doc.metadata.get('source', 'Unknown') for doc in context_docs]
            }
    
    def process_query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Process a query end-to-end.
        
        Args:
            query: User query
            k: Number of documents to retrieve
        
        Returns:
            Dictionary containing response and relevant documents
        """
        relevant_docs = self.query_documents(query, k=k)
        response = self.generate_response(query, relevant_docs)
        
        return {
            "response": response,
            "relevant_documents": relevant_docs
        }
    
    def index_documents(self, file_pattern: str = "**/*.xlsx") -> None:
        """
        Index documents end-to-end.
        
        Args:
            file_pattern: Glob pattern for files to index
        """
        documents = self.load_documents(file_pattern)
        splits = self.split_documents(documents)
        self.create_vectorstore(splits)