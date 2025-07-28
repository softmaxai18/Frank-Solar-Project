import os
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
import hashlib
import re
from langchain_ollama import ChatOllama 
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGFolderStructureDB:
    def __init__(self, knowledge_base_path: str, vector_db_path: str = "vector_db"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_db_path = vector_db_path
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)

        # Initialize LLM with better parameters
        self.llm = ChatOllama(
            model="gemma3:1b",
            temperature=0.1,  # Lower temperature for more factual responses
            num_ctx=4096,     # Larger context window
        )

        # Store database connections and file mappings
        self.db_connections = {}
        self.file_metadata = {}  # Store file metadata for better context
        
    def create_database_structure(self):
        """Create database structure mirroring folder structure"""
        logger.info("Creating database structure...")

        if os.path.exists(self.vector_db_path):
            existing_collections = self.chroma_client.list_collections()
            if existing_collections:
                logger.info(f"Found existing vector database with {len(existing_collections)} collections. Loading existing structure...")
                self._load_existing_databases()
                return
        
        # Walk through the knowledge base directory
        for root, dirs, files in os.walk(self.knowledge_base_path):
            relative_path = Path(root).relative_to(self.knowledge_base_path)
            
            # Skip the root directory itself
            if str(relative_path) == '.':
                continue
                
            # Create database for each folder
            db_name = self._sanitize_name(str(relative_path))
            db_path = f"{self.vector_db_path}/{db_name}.db"
            
            logger.info(f"Creating database: {db_name}")
            conn = sqlite3.connect(db_path)
            self.db_connections[db_name] = conn
            
            # Create tables for each txt file in the folder
            txt_files = [f for f in files if f.endswith('.txt')]
            for txt_file in txt_files:
                file_path = os.path.join(root, txt_file)
                self._create_table_for_file(conn, txt_file, file_path)
                
                # Store file metadata
                self.file_metadata[file_path] = {
                    'category': str(relative_path),
                    'filename': txt_file,
                    'db_name': db_name
                }
            
            conn.commit()
    
    def _load_existing_databases(self):
        """Load existing database connections"""
        db_files = Path(self.vector_db_path).glob("*.db")
        
        for db_file in db_files:
            db_name = db_file.stem
            logger.info(f"Loading existing database: {db_name}")
            conn = sqlite3.connect(str(db_file))
            self.db_connections[db_name] = conn
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"  Found {len(tables)} tables: {', '.join(tables)}")
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize folder/file names for database usage"""
        sanitized = re.sub(r'[^\w\-_\.]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized
    
    def _create_table_for_file(self, conn: sqlite3.Connection, filename: str, filepath: str):
        """Create table for individual file with enhanced metadata"""
        table_name = self._sanitize_name(filename.replace('.txt', ''))
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            content TEXT NOT NULL,
            content_hash TEXT,
            chunk_index INTEGER,
            file_path TEXT,
            file_size INTEGER,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        conn.execute(create_table_sql)
        logger.info(f"  Created table: {table_name}")
        
        self._process_and_store_file(conn, table_name, filepath)
    
    def _process_and_store_file(self, conn: sqlite3.Connection, table_name: str, filepath: str):
        """Process file content with enhanced chunking strategy"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            
            # Check if file is empty or too small
            if not content or len(content) < 10:
                logger.warning(f"    Skipping empty or too small file: {Path(filepath).name}")
                return
            
            # Enhanced chunking with semantic awareness
            chunks = self._smart_text_chunking(content)
            
            if not chunks:
                logger.warning(f"    No chunks generated from file: {Path(filepath).name}")
                return
            
            collection_name = f"{Path(filepath).parent.name}_{table_name}"
            collection = self._get_or_create_collection(collection_name)
            
            file_size = len(content)
            word_count = len(content.split())
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{table_name}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                content_hash = hashlib.md5(chunk.encode()).hexdigest()
                
                # Store in SQLite with enhanced metadata
                insert_sql = f"""
                INSERT OR REPLACE INTO {table_name} 
                (chunk_id, content, content_hash, chunk_index, file_path, file_size, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute(insert_sql, (chunk_id, chunk, content_hash, i, filepath, file_size, word_count))
                
                # Generate embedding
                embedding = self.embedding_model.embed_query(chunk)
                
                # Store in ChromaDB with enhanced metadata
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'file_path': filepath,
                        'table_name': table_name,
                        'chunk_index': i,
                        'content_hash': content_hash,
                        'category': Path(filepath).parent.name,
                        'filename': Path(filepath).name,
                        'file_size': file_size,
                        'word_count': word_count
                    }]
                )
            
            logger.info(f"    Processed {len(chunks)} chunks from {Path(filepath).name}")
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
    
    def _smart_text_chunking(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Enhanced text chunking with semantic awareness"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is too long, split by sentences
            if len(paragraph) > chunk_size:
                sentences = re.split(r'[.!?]+', paragraph)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(current_chunk) + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Keep some overlap
                            words = current_chunk.split()
                            if len(words) > overlap:
                                current_chunk = ' '.join(words[-overlap:]) + ' ' + sentence
                            else:
                                current_chunk = sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += ' ' + sentence if current_chunk else sentence
            else:
                # Normal paragraph handling
                if len(current_chunk) + len(paragraph) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Keep some overlap
                        words = current_chunk.split()
                        if len(words) > overlap:
                            current_chunk = ' '.join(words[-overlap:]) + ' ' + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk += '\n\n' + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_or_create_collection(self, collection_name: str):
        """Get or create ChromaDB collection"""
        sanitized_name = self._sanitize_name(collection_name).lower()
        try:
            collection = self.chroma_client.get_collection(sanitized_name)
        except:
            collection = self.chroma_client.create_collection(
                name=sanitized_name,
                metadata={"hnsw:space": "cosine"}
            )
        return collection
    
    def query_similar_content(self, query: str, n_results: int = 5, folder_filter: str = None, 
                            similarity_threshold: float = 0.3) -> List[Dict]:
        """Enhanced query with similarity threshold and better ranking"""
        query_embedding = self.embedding_model.embed_query(query)
        results = []
        
        collections = self.chroma_client.list_collections()
        
        for collection_info in collections:
            collection_name = collection_info.name
            
            if folder_filter and not collection_name.startswith(folder_filter.lower()):
                continue
            
            collection = self.chroma_client.get_collection(collection_name)
            
            if collection.count() == 0:
                continue
                
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, collection.count()),  # Get more results for filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            for i in range(len(query_results['documents'][0])):
                similarity = 1 - query_results['distances'][0][i]
                
                # Filter by similarity threshold
                if similarity >= similarity_threshold:
                    results.append({
                        'collection': collection_name,
                        'document': query_results['documents'][0][i],
                        'metadata': query_results['metadatas'][0][i],
                        'distance': query_results['distances'][0][i],
                        'similarity': similarity
                    })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:n_results]
    
    def get_most_relevant_file_content(self, query: str, similarity_threshold: float = 0.4) -> Optional[Dict]:
        """Get the most relevant file content with enhanced relevance scoring"""
        results = self.query_similar_content(query, n_results=10, similarity_threshold=similarity_threshold)
        
        if not results:
            return None
        
        # Group results by file and calculate aggregated relevance
        file_scores = {}
        
        for result in results:
            file_path = result['metadata']['file_path']
            similarity = result['similarity']
            
            if file_path not in file_scores:
                file_scores[file_path] = {
                    'total_score': 0,
                    'chunk_count': 0,
                    'max_similarity': 0,
                    'metadata': result['metadata']
                }
            
            file_scores[file_path]['total_score'] += similarity
            file_scores[file_path]['chunk_count'] += 1
            file_scores[file_path]['max_similarity'] = max(
                file_scores[file_path]['max_similarity'], similarity
            )
        
        # Calculate final relevance score (weighted average + max similarity)
        best_file = None
        best_score = 0
        
        for file_path, scores in file_scores.items():
            avg_score = scores['total_score'] / scores['chunk_count']
            final_score = (avg_score * 0.7) + (scores['max_similarity'] * 0.3)
            
            if final_score > best_score:
                best_score = final_score
                best_file = file_path
        
        if not best_file:
            return None
        
        # Read the full file content
        try:
            with open(best_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return {
                'file_path': best_file,
                'content': content,
                'similarity': best_score,
                'metadata': file_scores[best_file]['metadata'],
                'relevant_chunks': file_scores[best_file]['chunk_count']
            }
        except Exception as e:
            logger.error(f"Error reading file {best_file}: {str(e)}")
            return None
    
    def get_contextual_response(self, query: str, context_limit: int = 3):
        """Generate response using multiple relevant sources with context"""
        # Get multiple relevant chunks for context
        similar_docs = self.query_similar_content(query, n_results=context_limit, similarity_threshold=0.3)
        
        if not similar_docs:
            yield "I don't have relevant information in my knowledge base to answer your question. Please try rephrasing your question or ask about water heater topics like types, installation, maintenance, or energy efficiency."
            return
        
        # Build rich context with source information
        context_parts = []
        sources_used = set()
        
        for i, doc in enumerate(similar_docs, 1):
            source_file = Path(doc['metadata']['file_path']).name
            category = doc['metadata'].get('category', 'Unknown')
            similarity = doc['similarity']
            
            if source_file not in sources_used:
                sources_used.add(source_file)
                context_parts.append(
                    f"[Source {i}: {source_file} from {category} (Relevance: {similarity:.2f})]\n{doc['document']}"
                )
        
        context = "\n\n".join(context_parts)
        
        # Create comprehensive prompt
        system_prompt = """You are an expert water heater consultant with access to comprehensive technical documentation. 
        Your role is to provide accurate, helpful, and detailed answers about water heaters based on the provided context.
        
        Guidelines:
        - Answer only based on the provided context
        - If the context doesn't contain sufficient information, clearly state this
        - Cite specific sources when making claims
        - Provide practical, actionable advice when applicable
        - Be comprehensive but concise
        - Use technical terms appropriately but explain them when necessary"""
        
        user_prompt = f"""Context Information:
            {context}

            User Question: {query}

            Please provide a comprehensive answer based on the context above. If you reference specific information, mention which source it comes from.
        """

        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]
        
        try:
            for chunk in self.llm.stream(messages):
                yield chunk
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def get_full_file_response(self, query: str):
        """Get response based on the most relevant complete file"""
        file_content = self.get_most_relevant_file_content(query)
        
        if not file_content:
            yield "I couldn't find a relevant file in my knowledge base to answer your question. Please try rephrasing your question or ask about specific water heater topics."
            return
        
        file_path = file_content['file_path']
        content = file_content['content']
        similarity = file_content['similarity']
        category = file_content['metadata'].get('category', 'Unknown')
        
        system_prompt = """You are an expert water heater consultant. You have access to a complete technical document that is highly relevant to the user's question.

        Guidelines:
        - Base your answer entirely on the provided document
        - Be thorough and comprehensive
        - Organize information logically
        - Include specific details, numbers, and recommendations when available
        - If the document doesn't fully answer the question, clearly state what information is missing
        - Mention the source document for credibility"""
        
        user_prompt = f"""Complete Document Content:
            Source: {Path(file_path).name} (Category: {category}, Relevance: {similarity:.2f})

            {content}

            User Question: {query}

            Please provide a comprehensive answer based on this document. Reference specific information from the document and organize your response clearly."""

        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]
        
        try:
            for chunk in self.llm.stream(messages):
                yield chunk
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def get_database_structure(self) -> Dict:
        """Get the complete database structure with enhanced metadata"""
        structure = {}
        
        for db_name, conn in self.db_connections.items():
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            structure[db_name] = {}
            for table in tables:
                # Check if enhanced columns exist
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'word_count' in columns and 'file_size' in columns:
                    cursor.execute(f"SELECT COUNT(*), AVG(word_count), MAX(file_size) FROM {table}")
                    count, avg_words, max_size = cursor.fetchone()
                    structure[db_name][table] = {
                        'chunk_count': count,
                        'avg_words_per_chunk': avg_words or 0,
                        'file_size': max_size or 0
                    }
                else:
                    # Fallback for old database structure
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    structure[db_name][table] = {
                        'chunk_count': count,
                        'avg_words_per_chunk': 0,
                        'file_size': 0
                    }
        
        return structure   
    
    def close_connections(self):
        """Close all database connections"""
        for conn in self.db_connections.values():
            conn.close()
        logger.info("All database connections closed.")


def main():
    """Main function with enhanced user interaction"""
    rag_system = EnhancedRAGFolderStructureDB("Knowledge_Base")
    
    try:
        # Create database structure
        rag_system.create_database_structure()
        
        print("\n" + "="*60)
        print("🔥 ENHANCED WATER HEATER KNOWLEDGE BASE 🔥")
        print("="*60)
        print("Available response modes:")
        print("1. 'full' - Get response from most relevant complete file")
        print("2. 'context' - Get response from multiple relevant sources") 
        print("3. 'structure' - View database structure")
        print("Type 'exit' or 'quit' to exit")
        print("="*60)
        
        while True:
            print("\n" + "-"*40)
            user_input = input("Enter your query: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Thank you for using the Water Heater Knowledge Base!")
                break
            
            if user_input.lower() == 'structure':
                structure = rag_system.get_database_structure()
                print("\n📊 DATABASE STRUCTURE:")
                for db_name, tables in structure.items():
                    print(f"\n📁 {db_name}:")
                    for table_name, info in tables.items():
                        print(f"  📄 {table_name}: {info['chunk_count']} chunks, "
                              f"avg {info['avg_words_per_chunk']:.1f} words/chunk")
                continue
            
            if not user_input:
                print("⚠️ Please enter a valid query.")
                continue
            
            # Ask for response mode
            print("\nResponse mode:")
            print("1. Full file analysis (recommended)")
            print("2. Multi-source context")
            
            mode = input("Choose mode (1 or 2, default=1): ").strip()
            
            print(f"\n🔍 Processing query: '{user_input}'")
            print("💭 Generating response...\n")
            
            try:
                if mode == '2':
                    response_generator = rag_system.get_contextual_response(user_input)
                else:
                    response_generator = rag_system.get_full_file_response(user_input)
                
                print("📋 Response:")
                print("-" * 40)
                
                for chunk in response_generator:
                    print(chunk.content, end="", flush=True)
                
                print("\n" + "-" * 40)
                
            except Exception as e:
                print(f"❌ Error generating response: {str(e)}")
                logger.error(f"Error in main loop: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Error in main: {str(e)}")
    finally:
        rag_system.close_connections()


if __name__ == "__main__":
    main()