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
from datetime import datetime

class ConversationalRAGBot:
    def __init__(self, knowledge_base_path: str, vector_db_path: str = "vector_db", domain: str = "water_heaters"):
        # Initialize RAG system
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_db_path = vector_db_path
        self.domain = domain
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        
        # LLM for conversation
        self.llm = ChatOllama(model="gemma3:1b")
        
        # Store database connections
        self.db_connections = {}
        
        # Conversation state
        self.conversation_history = []
        self.user_profile = {}
        self.current_stage = "introduction"
        self.questions_asked = set()
        
        # Domain-specific question templates
        self.question_templates = self._load_question_templates()
        
        # Initialize the RAG system
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the RAG system from your existing code"""
        print("Initializing RAG system...")
        
        if os.path.exists(self.vector_db_path):
            existing_collections = self.chroma_client.list_collections()
            if existing_collections:
                print(f"Found existing vector database with {len(existing_collections)} collections.")
                self._load_existing_databases()
                return
        
        # Create database structure if it doesn't exist
        self._create_database_structure()
    
    def _load_existing_databases(self):
        """Load existing database connections"""
        db_files = Path(self.vector_db_path).glob("*.db")
        
        for db_file in db_files:
            db_name = db_file.stem
            print(f"Loading existing database: {db_name}")
            conn = sqlite3.connect(str(db_file))
            self.db_connections[db_name] = conn
    
    def _create_database_structure(self):
        """Create database structure mirroring folder structure"""
        print("Creating database structure...")
        
        for root, dirs, files in os.walk(self.knowledge_base_path):
            relative_path = Path(root).relative_to(self.knowledge_base_path)
            
            if str(relative_path) == '.':
                continue
                
            db_name = self._sanitize_name(str(relative_path))
            db_path = f"{self.vector_db_path}/{db_name}.db"
            
            conn = sqlite3.connect(db_path)
            self.db_connections[db_name] = conn
            
            txt_files = [f for f in files if f.endswith('.txt')]
            for txt_file in txt_files:
                self._create_table_for_file(conn, txt_file, os.path.join(root, txt_file))
            
            conn.commit()
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize folder/file names for database usage"""
        sanitized = re.sub(r'[^\w\-_\.]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized
    
    def _create_table_for_file(self, conn: sqlite3.Connection, filename: str, filepath: str):
        """Create table for individual file with vector storage"""
        table_name = self._sanitize_name(filename.replace('.txt', ''))
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            content TEXT NOT NULL,
            content_hash TEXT,
            chunk_index INTEGER,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        conn.execute(create_table_sql)
        self._process_and_store_file(conn, table_name, filepath)
    
    def _process_and_store_file(self, conn: sqlite3.Connection, table_name: str, filepath: str):
        """Process file content and store in database with vectors"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunks = self._split_text_into_chunks(content)
            
            collection_name = f"{Path(filepath).parent.name}_{table_name}"
            collection = self._get_or_create_collection(collection_name)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{table_name}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                content_hash = hashlib.md5(chunk.encode()).hexdigest()
                
                insert_sql = f"""
                INSERT OR REPLACE INTO {table_name} 
                (chunk_id, content, content_hash, chunk_index, file_path)
                VALUES (?, ?, ?, ?, ?)
                """
                conn.execute(insert_sql, (chunk_id, chunk, content_hash, i, filepath))
                
                embedding = self.embedding_model.embed_query(chunk)
                
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'file_path': filepath,
                        'table_name': table_name,
                        'chunk_index': i,
                        'content_hash': content_hash
                    }]
                )
            
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size - 100:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
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
    
    def _load_question_templates(self) -> Dict:
        """Load domain-specific question templates"""
        if self.domain == "water_heaters":
            return {
                "introduction": [
                    "What is your main reason for looking into a new water heater?",
                    "Are you replacing an existing water heater or installing a new one?",
                    "What type of water heater do you currently have?"
                ],
                "current_setup": [
                    "How old is your current water heater?",
                    "How many people does the unit serve in your house?",
                    "What appliances do you have that use hot water?",
                    "Where is your water heater located?",
                    "Do you have natural gas available in your area?"
                ],
                "satisfaction": [
                    "How satisfied are you with your current unit?",
                    "Have you experienced any issues with hot water availability?",
                    "Do you perform regular maintenance on your water heater?"
                ],
                "preferences": [
                    "Would you be interested in a different type of water heater?",
                    "Are you willing to spend more upfront for long-term savings?",
                    "How important are energy efficiency and environmental impact to you?"
                ],
                "technical": [
                    "What is your available installation space like?",
                    "Are you interested in smart/connected features?",
                    "Do you have any electrical or plumbing constraints?"
                ]
            }
        else:
            # Generic templates for other domains
            return {
                "introduction": [
                    "What brings you here today?",
                    "What are you looking to accomplish?",
                    "Can you tell me about your current situation?"
                ]
            }
    
    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query the knowledge base for relevant information"""
        query_embedding = self.embedding_model.embed_query(query)
        results = []
        
        collections = self.chroma_client.list_collections()
        
        for collection_info in collections:
            collection_name = collection_info.name
            collection = self.chroma_client.get_collection(collection_name)
            
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            for i in range(len(query_results['documents'][0])):
                results.append({
                    'collection': collection_name,
                    'document': query_results['documents'][0][i],
                    'metadata': query_results['metadatas'][0][i],
                    'distance': query_results['distances'][0][i],
                    'similarity': 1 - query_results['distances'][0][i]
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:n_results]
    
    def analyze_user_response(self, response: str) -> Dict:
        """Analyze user response to extract key information"""
        # Use LLM to extract structured information from user response
        analysis_prompt = f"""
        Analyze this user response about water heaters and extract key information:
        
        User Response: "{response}"
        
        Extract the following information if mentioned:
        - Water heater type (electric, gas, tankless, heat pump, etc.)
        - Age/vintage of current unit
        - Number of people in household
        - Location/space constraints
        - Budget considerations
        - Efficiency preferences
        - Problems with current unit
        - Special requirements
        
        Respond in JSON format with extracted information. Use null for missing information.
        """
        
        messages = [
            ("system", "You are an expert at extracting structured information from conversational responses about water heaters. Always respond with valid JSON."),
            ("human", analysis_prompt)
        ]
        
        try:
            llm_response = ""
            for chunk in self.llm.stream(messages):
                llm_response += chunk.content
            
            # Try to extract JSON from response
            import json
            # Find JSON in the response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = llm_response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback: simple keyword extraction
        response_lower = response.lower()
        extracted_info = {}
        
        # Water heater types
        if any(word in response_lower for word in ['tankless', 'on-demand']):
            extracted_info['water_heater_type'] = 'tankless'
        elif 'electric' in response_lower:
            extracted_info['water_heater_type'] = 'electric'
        elif 'gas' in response_lower:
            extracted_info['water_heater_type'] = 'gas'
        
        # Extract numbers (could be age, people, gallons)
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            extracted_info['mentioned_numbers'] = [int(n) for n in numbers]
        
        return extracted_info
    
    def determine_next_question(self, user_response: str) -> str:
        """Determine the next question based on user response and knowledge base"""
        # Analyze the user response
        extracted_info = self.analyze_user_response(user_response)
        
        # Update user profile with extracted information
        self.user_profile.update(extracted_info)
        
        # Query knowledge base for relevant information
        relevant_info = self.query_knowledge_base(user_response, n_results=2)
        
        # Determine conversation stage and next question
        next_question = self._generate_contextual_question(user_response, extracted_info, relevant_info)
        
        return next_question
    
    def _generate_contextual_question(self, user_response: str, extracted_info: Dict, relevant_info: List[Dict]) -> str:
        """Generate contextual follow-up question using LLM"""
        # Build context from knowledge base
        knowledge_context = ""
        if relevant_info:
            knowledge_context = "\n\n".join([
                f"Knowledge: {doc['document']}"
                for doc in relevant_info[:2]
            ])
        
        # Build conversation context
        conversation_context = "\n".join([
            f"{'Bot' if entry['speaker'] == 'bot' else 'User'}: {entry['message']}"
            for entry in self.conversation_history[-3:]  # Last 3 exchanges
        ])
        
        question_generation_prompt = f"""
        You are conducting a consultative interview about water heaters. Based on the conversation so far and your knowledge, ask the next most logical question.
        
        Previous Conversation:
        {conversation_context}
        
        User's Latest Response: "{user_response}"
        
        Extracted Information: {extracted_info}
        
        Relevant Knowledge:
        {knowledge_context}
        
        Guidelines:
        1. Ask one focused question at a time
        2. Build on what the user has already shared
        3. Move logically from basic to more specific information
        4. Use the knowledge base to ask informed questions
        5. Don't repeat questions already asked
        6. Be conversational and helpful
        
        Questions already asked: {list(self.questions_asked)}
        
        Ask the next most appropriate question:
        """
        
        messages = [
            ("system", "You are an expert water heater consultant who asks thoughtful, logical follow-up questions. Keep questions conversational and focused."),
            ("human", question_generation_prompt)
        ]
        
        try:
            llm_response = ""
            for chunk in self.llm.stream(messages):
                llm_response += chunk.content
            
            # Clean up the response to extract just the question
            question = llm_response.strip()
            
            # Remove any prefixes like "Question:" or "Next question:"
            prefixes = ["question:", "next question:", "follow-up:", "ask:"]
            for prefix in prefixes:
                if question.lower().startswith(prefix):
                    question = question[len(prefix):].strip()
            
            return question
            
        except Exception as e:
            print(f"Error generating question: {e}")
            # Fallback to template questions
            return self._get_fallback_question()
    
    def _get_fallback_question(self) -> str:
        """Get fallback question from templates"""
        # Determine current stage based on conversation history
        if len(self.conversation_history) < 2:
            return self.question_templates["introduction"][0]
        elif len(self.conversation_history) < 6:
            available_questions = [q for q in self.question_templates["current_setup"] if q not in self.questions_asked]
            return available_questions[0] if available_questions else "Is there anything else you'd like me to know about your situation?"
        else:
            available_questions = [q for q in self.question_templates["preferences"] if q not in self.questions_asked]
            return available_questions[0] if available_questions else "Based on what you've told me, let me find some recommendations for you."
    
    def start_conversation(self):
        """Start the conversational interview"""
        print("=== Water Heater Consultation Bot ===")
        print("Hi! I'm here to help you find the perfect water heater solution.")
        print("I'll ask you a few questions to understand your needs better.\n")
        
        # Start with first question
        first_question = self.question_templates["introduction"][0]
        print(f"Bot: {first_question}")
        self.questions_asked.add(first_question)
        self.conversation_history.append({
            "speaker": "bot",
            "message": first_question,
            "timestamp": datetime.now()
        })
        
        return first_question
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Add user response to conversation history
        self.conversation_history.append({
            "speaker": "user",
            "message": user_input,
            "timestamp": datetime.now()
        })
        
        # Check if user wants to end conversation
        if any(phrase in user_input.lower() for phrase in ['goodbye', 'bye', 'thanks', 'that\'s all', 'no more questions']):
            return self._generate_final_recommendations()
        
        # Determine next question
        next_question = self.determine_next_question(user_input)
        
        # Add bot response to conversation history
        self.conversation_history.append({
            "speaker": "bot",
            "message": next_question,
            "timestamp": datetime.now()
        })
        
        self.questions_asked.add(next_question)
        
        return next_question
    
    def _generate_final_recommendations(self) -> str:
        """Generate final recommendations based on the conversation"""
        # Summarize the conversation for recommendation generation
        conversation_summary = " ".join([
            entry['message'] for entry in self.conversation_history 
            if entry['speaker'] == 'user'
        ])
        
        # Query knowledge base for recommendations
        relevant_info = self.query_knowledge_base(conversation_summary, n_results=5)
        
        # Build knowledge context
        knowledge_context = "\n\n".join([
            f"Source: {doc['metadata']['file_path']}\n{doc['document']}"
            for doc in relevant_info
        ])
        
        recommendation_prompt = f"""
        Based on this water heater consultation conversation, provide personalized recommendations:
        
        User Profile: {self.user_profile}
        
        Conversation Summary: {conversation_summary}
        
        Relevant Knowledge:
        {knowledge_context}
        
        Provide specific recommendations including:
        1. Recommended water heater type(s)
        2. Key features to look for
        3. Estimated costs or savings
        4. Installation considerations
        5. Next steps
        
        Be specific and actionable based on their situation.
        """
        
        messages = [
            ("system", "You are a water heater expert providing personalized recommendations based on consultation."),
            ("human", recommendation_prompt)
        ]
        
        print("Bot: Thank you for all that information! Let me provide some personalized recommendations based on our conversation:\n")
        
        try:
            for chunk in self.llm.stream(messages):
                print(chunk.content, end="", flush=True)
            print("\n\nFeel free to ask if you have any questions about these recommendations!")
        except Exception as e:
            print(f"I'd be happy to provide recommendations, but I'm having trouble generating them right now. Error: {e}")
        
        return "Recommendations provided"
    
    def run_interactive_session(self):
        """Run interactive conversation session"""
        self.start_conversation()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    print("Bot: Goodbye! Feel free to come back anytime you need help with water heaters.")
                    break
                
                response = self.process_user_input(user_input)
                print(f"Bot: {response}")
                
                # If we've had a substantial conversation, offer to wrap up
                if len(self.conversation_history) > 12:
                    print("\n(Type 'recommendations' to get final recommendations or continue with more questions)")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye! Feel free to come back anytime.")
                break
            except Exception as e:
                print(f"Bot: I'm sorry, I encountered an error: {e}")
                print("Let's continue with our conversation.")
    
    def close_connections(self):
        """Close all database connections"""
        for conn in self.db_connections.values():
            conn.close()

# Usage Example
def main():
    # Initialize the conversational bot
    bot = ConversationalRAGBot("Knowledge_Base")
    
    try:
        # Run interactive session
        bot.run_interactive_session()
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close connections
        bot.close_connections()

if __name__ == "__main__":
    main()