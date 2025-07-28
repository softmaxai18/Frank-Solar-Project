import os

from pathlib import Path
from typing import List, Dict, Optional

from langchain_openai import OpenAIEmbeddings

import chromadb
from chromadb.config import Settings
import hashlib
import re
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGFolderStructureDB:
    def __init__(self, knowledge_base_path: str, vector_db_path: str = "vector_db"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_db_path = vector_db_path
        self.embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)

        # Initialize LLM with better parameters
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            # num_ctx=4096,     # Larger context window
        )

        # Store database connections and file mappings
        self.file_metadata = {}  # Store file metadata for better context
        
    def create_database_structure(self):
        # Check if Chroma already has any collections
        existing_collections = self.chroma_client.list_collections()
        if existing_collections :
            logger.info("✅ Vector DB already populated. Skipping embedding.")
            return

        logger.info("🛠️ Creating vector DB structure from knowledge base...")

        for root, dirs, files in os.walk(self.knowledge_base_path):
            relative_path = Path(root).relative_to(self.knowledge_base_path)
            if str(relative_path) == '.':
                continue

            txt_files = [f for f in files if f.endswith('.txt')]
            for txt_file in txt_files:
                file_path = os.path.join(root, txt_file)
                self._embed_and_store_in_chroma(txt_file, file_path, str(relative_path))


    def _embed_and_store_in_chroma(self, filename: str, filepath: str, category: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()

            if not content or len(content) < 10:
                logger.warning(f"Skipping empty/small file: {filename}")
                return

            chunks = self._smart_text_chunking(content)
            if not chunks:
                return

            table_name = self._sanitize_name(filename.replace('.txt', ''))
            collection_name = f"{category}_{table_name}"
            collection = self._get_or_create_collection(collection_name)

            # Get existing IDs
            existing_ids = set()
            try:
                existing_docs = collection.get(include=["ids"])
                existing_ids.update(existing_docs["ids"])
            except:
                pass  # Collection may be new

            file_size = len(content)
            word_count = len(content.split())

            new_chunks = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{table_name}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                if chunk_id in existing_ids:
                    continue  # Skip already embedded chunk

                content_hash = hashlib.md5(chunk.encode()).hexdigest()
                embedding = self.embedding_model.embed_query(chunk)

                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'file_path': filepath,
                        'table_name': table_name,
                        'chunk_index': i,
                        'content_hash': content_hash,
                        'category': category,
                        'filename': filename,
                        'file_size': file_size,
                        'word_count': word_count
                    }]
                )
                new_chunks += 1

            logger.info(f"Embedded {new_chunks} new chunks from {filename}")
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")


    def _sanitize_name(self, name: str) -> str:
        """Sanitize folder/file names for database usage"""
        sanitized = re.sub(r'[^\w\-_\.]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized
    
 
    def _smart_text_chunking(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
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
            # Convert the sync generator to async
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
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
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    


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
        print("Type 'exit' or 'quit' to exit")
        print("="*60)
        
        while True:
            print("\n" + "-"*40)
            user_input = input("Enter your query: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Thank you for using the Water Heater Knowledge Base!")
                break
            
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
                    print(chunk, end="", flush=True)  # ✅ Fix here
                
                print("\n" + "-" * 40)
                
            except Exception as e:
                print(f"❌ Error generating response: {str(e)}")
                logger.error(f"Error in main loop: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()