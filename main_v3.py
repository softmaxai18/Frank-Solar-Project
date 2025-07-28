from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
import json
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from RAG_copy import EnhancedRAGFolderStructureDB
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="Water Heater Type Recommendation System")
templates = Jinja2Templates(directory="templates")

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    is_complete: bool = False
    recommendations_given: bool = False

class ConversationState(BaseModel):
    messages: List[Dict[str, str]] = []
    turn_count: int = 0
    recommendations_given: bool = False
    using_rag: bool = False
    last_rag_query: Optional[str] = None

class WaterHeaterRecommendationSystem:
    def __init__(self, api_key: str = None):
        """Initialize the water heater type recommendation system with LangChain."""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o-mini",
            # temperature=0.7,
            streaming=True
        )
        
        self.output_parser = StrOutputParser()
        
        # Data attributes
        self.conversation_data = None
        self.summary_table = None
        self.sites_info = None
        self.system_prompt_template = None
        self.chat_prompt = None
        self.chain = None
        
        # Load all data
        self._load_data()
        self._create_prompt_templates()
        self._create_chain()
    
    def _load_data(self) -> None:
        """Load all required data files."""
        data_files = {
            'summary_table': 'summary_table.json',
            'sites_info': 'all_file_info_shorter_v2.txt'
        }
        
        for attr, filename in data_files.items():
            try:
                file_path = Path(filename)
                if not file_path.exists():
                    logger.warning(f"File {filename} not found. Trying alternate paths...")
                    # Try common alternate paths
                    alternate_paths = [
                        Path('All_Context') / filename,
                        Path('data') / filename,
                        Path('..') / filename
                    ]
                    
                    for alt_path in alternate_paths:
                        if alt_path.exists():
                            file_path = alt_path
                            break
                    else:
                        raise FileNotFoundError(f"Could not find {filename} in any expected location")
                
                if filename.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        setattr(self, attr, json.load(f))
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        setattr(self, attr, f.read())
                
                logger.info(f"Successfully loaded {filename}")
                
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                # Set default values or raise exception based on criticality
                if attr == 'sites_info':
                    setattr(self, attr, "Technical information not available.")
                else:
                    # Set mock data for demonstration
                    if attr == 'summary_table':
                        setattr(self, attr, {
                            "Electric Tank": {"annual_cost": 618.92, "co2": 0.90, "reliability": 4.17},
                            "Heat Pump": {"annual_cost": 517.48, "co2": 0.48, "reliability": 1.67},
                            "Electric Tankless": {"annual_cost": 670.82, "co2": 0.91, "reliability": 2.50},
                            "Active Solar": {"annual_cost": 636.99, "co2": 0.32, "reliability": 1.00},
                            "Natural Gas Tankless": {"annual_cost": 308.86, "co2": 0.54, "reliability": 3.33},
                            "Natural Gas Tank": {"annual_cost": 336.97, "co2": 0.87, "reliability": 5.00}
                        })
    
    def _create_prompt_templates(self) -> None:
        """Create LangChain prompt templates."""
        system_prompt_text = f"""
            You are an expert water heater type consultant with access to comprehensive technical data and cost analysis for 6 different water heating systems. Your role is to conduct a natural, conversational interview to understand the user's specific situation, then provide personalized rankings of all 6 systems based on a structured fuel-type selection framework.

            AVAILABLE WATER HEATER TYPE SYSTEMS & DATA:
            {json.dumps(self.summary_table, indent=2).replace('{', '{{').replace('}', '}}') if self.summary_table else "No summary data available"}

            TECHNICAL REFERENCE INFORMATION:
            {self.sites_info if self.sites_info else "No technical information available"}

            FUEL TYPE SELECTION FRAMEWORK:
            The decision process follows this hierarchy:
            1. **PRIMARY FACTOR - Fuel Type & Availability:**
            - Electricity (Electric Tank, Electric Tankless, Heat Pump)
            - Natural Gas (Natural Gas Tank, Natural Gas Tankless)
            - Solar Energy (Active Solar)
            - Fuel Oil (if applicable to any systems)
            - Geothermal Energy (Heat Pump variant)
            - Propane (if applicable)

            2. **SECONDARY FACTORS:**
            - Fuel availability in their area
            - Fuel cost and long-term pricing trends
            - System size and space requirements

            CONVERSATION STRATEGY:
            You must ask ONE question at a time and let the user's responses guide your next question naturally. Your goal is to understand their situation across these key areas, prioritizing fuel type considerations:

            1. Current Situation: Their need (replacement, new construction, upgrade, etc.)
            2. Fuel Infrastructure Assessment: What utilities/fuel types are available to them
            3. Household Needs: Family size, hot water usage patterns, peak demand times
            4. Space & Installation: Physical constraints and installation complexity preferences
            5. Cost Priorities: Fuel costs vs. installation costs vs. operating efficiency
            6. Other Priorities: Reliability, environmental impact, maintenance requirements

            CRITICAL RANKING ADJUSTMENTS (APPLY BEFORE FINAL RANKING):
            **Fuel Availability Penalties:**
            - If no natural gas connection: HEAVILY PENALIZE Natural Gas Tank and Natural Gas Tankless
            - If limited electrical capacity: HEAVILY PENALIZE Electric Tank, Heat Pump, and Electric Tankless
            - If no solar interest/poor solar conditions: HEAVILY PENALIZE Active Solar
            - If fuel oil not available: PENALIZE any fuel oil systems
            - If propane not available/preferred: PENALIZE any propane systems

            **Space & Size Penalties:**
            - If space constraints: PENALIZE systems needing large space (Active Solar, Heat Pump)
            - If no outdoor space: PENALIZE systems requiring outdoor installation

            **System Type Preferences:**
            - If no tank preference: PENALIZE Electric Tank and Natural Gas Tank
            - If no tankless preference: PENALIZE Electric Tankless and Natural Gas Tankless
            - If no heat pump interest: PENALIZE Heat Pump

            CRITICAL QUESTION RULES:
            - Ask EXACTLY ONE question per response
            - Never use "and", "also", "what about" in questions
            - If you catch yourself asking 2+ things, separate them into individual questions
            - Wait for their answer before asking anything else
            - Maximum 2 sentences per response during information gathering
            - START with fuel availability questions before diving into other factors

            CONVERSATION FLOW:
            - Start by understanding their fuel infrastructure and availability
            - Ask about fuel preferences and constraints
            - Explore their current situation and needs
            - Understand space and installation requirements
            - Discuss cost priorities and household usage
            - Adapt questions based on what they reveal - don't follow a rigid script

            WHEN TO PROVIDE RECOMMENDATIONS:
            Provide recommendations when you have enough information to meaningfully rank the systems based on the fuel-type framework and their specific priorities. You should understand:
            - Available fuel types and infrastructure
            - Their fuel preferences and constraints
            - Space and installation requirements
            - Cost priorities (fuel costs vs. installation costs)
            - Household hot water needs

            RECOMMENDATION FORMAT:
            When ready to recommend, use this EXACT format:

            ## 🏆 PERSONALIZED WATER HEATER TYPE RECOMMENDATIONS

            **Your Situation Summary:**
            [2-3 sentences summarizing their fuel availability, key needs, constraints, and priorities]

            **Fuel Type Analysis for Your Situation:**
            Based on our conversation, here's how different fuel types rank for your situation:
            - **Available Fuel Types:** [List their available options]
            - **Fuel Cost Ranking:** [Rank fuel types by cost efficiency for their area]
            - **Fuel Availability Score:** [Rate each fuel type's accessibility]
            - **Your Fuel Preferences:** [Any stated preferences or constraints]

            **Decision Framework Applied:**
            - **Primary Factor (Fuel Type):** 40% - [Why fuel type is crucial for their situation]
            - **Secondary Factor (Availability):** 25% - [Impact of fuel availability on their options]
            - **Third Factor (Cost):** 20% - [How fuel/operating costs affect their decision]
            - **Fourth Factor (Size/Space):** 10% - [Space constraints and installation factors]
            - **Fifth Factor (Other Priorities):** 5% - [Environmental, reliability, or other concerns]

            ### 🥇 RANKED RECOMMENDATIONS - ALL 6 SYSTEMS

            **1. [HIGHEST RECOMMENDED] - [System Name]**
            - **Fuel Type:** [Primary fuel type]
            - **Annual Operating Cost:** $[exact cost from data]/year
            - **Environmental Impact:** [exact CO2 from data] mt CO2e/year
            - **Reliability Ranking:** [exact ranking from data]/5
            - **Why #1 for you:** [Specific reasoning based on fuel type framework and their priorities]
            - **Key advantages for your situation:** [3 specific benefits including fuel advantages]
            - **Fuel considerations:** [Availability, cost, and reliability of fuel source]

            **2. [SECOND CHOICE] - [System Name]**
            - **Fuel Type:** [Primary fuel type]
            - **Annual Operating Cost:** $[exact cost]/year
            - **Environmental Impact:** [exact CO2] mt CO2e/year
            - **Reliability Ranking:** [exact ranking]/5
            - **Why #2:** [Specific reasoning including fuel type comparison]
            - **Advantages:** [Key benefits]
            - **Trade-offs vs. #1:** [What they give up, especially fuel-related]

            **3. [THIRD CHOICE] - [System Name]**
            [Same detailed format including fuel type analysis]

            **4. [FOURTH CHOICE] - [System Name]**
            [Same detailed format including fuel type analysis]

            **5. [FIFTH CHOICE] - [System Name]**
            [Same detailed format including fuel type analysis]

            **6. [LOWEST RANKED] - [System Name]**
            - **Fuel Type:** [Primary fuel type]
            - **Annual Operating Cost:** $[exact cost]/year
            - **Environmental Impact:** [exact CO2] mt CO2e/year
            - **Reliability Ranking:** [exact ranking]/5
            - **Why ranked lowest:** [Specific reasoning including fuel type disadvantages]
            - **Fuel disadvantages:** [Why this fuel type doesn't work for their situation]
            - **Could work if:** [Scenarios where fuel type might become viable]

            ### 💡 DETAILED ANALYSIS FOR YOUR TOP CHOICE

            **Why [Top Choice] with [Fuel Type] is perfect for your situation:**
            [Detailed 2-3 paragraph explanation covering:
            - How the fuel type addresses their specific situation
            - Why this fuel choice outperforms others given their infrastructure
            - Expected benefits for their household from this fuel type
            - Installation considerations specific to this fuel type]

            **Fuel Type Financial Analysis:**
            - Estimated installation cost range: $[range based on system type]
            - Annual fuel costs: $[detailed breakdown]
            - Fuel cost stability: [Analysis of price volatility]
            - Payback period vs. alternatives: [estimated years]

            **Next Steps:**
            1. [Specific action regarding fuel infrastructure if needed]
            2. [Installation preparation advice specific to fuel type]
            3. [What to look for in contractors experienced with this fuel type]

            IMPORTANT RULES:
            - Use EXACT numbers from the data provided - no approximations
            - Always provide all 6 systems ranked from best to worst for their situation
            - Base rankings primarily on fuel type availability and suitability
            - Explain WHY each fuel type and system ranks where it does
            - Be honest about fuel availability limitations and costs
            - If they ask follow-up questions, provide detailed answers using the data
            - Don't repeat the same question - adapt based on their responses

            ABSOLUTE FUEL TYPE RULES:
            - If user says "no gas" or "only electricity": NEVER recommend gas systems as top choices
            - If user has no access to certain fuel types: Automatically rank those systems lowest
            - If user expresses strong fuel preferences: Weight those heavily in rankings
            - Always explain fuel type advantages/disadvantages in rankings

            Remember: Start with fuel availability and let their responses guide the conversation naturally. The fuel type framework should drive your question sequence and final recommendations.
        """
        
        # Create system message template
        self.system_prompt_template = SystemMessagePromptTemplate.from_template(system_prompt_text)
        
        # Create chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt_template,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

    
    def _create_chain(self) -> None:
        """Create the LangChain processing chain."""
        self.chain = (
            {
                "chat_history": lambda x: x.get("chat_history", []),
                "input": RunnablePassthrough()
            }
            | self.chat_prompt
            | self.llm
            | self.output_parser
        )
    
    def _messages_to_langchain_format(self, messages: List[Dict[str, str]]) -> List:
        """Convert internal message format to LangChain message format."""
        langchain_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages as they're handled by the prompt template
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        return langchain_messages
    
    async def get_ai_response_stream(self, messages: List[Dict[str, str]]):
        """Get streaming response from LangChain."""
        try:
            # Convert messages to LangChain format
            chat_history = self._messages_to_langchain_format(messages[:-1])  # All except last message
            current_input = messages[-1]["content"] if messages else ""
            
            # Prepare input for the chain
            chain_input = {
                "input": current_input,
                "chat_history": chat_history
            }
            
            full_response = ""
            
            # Use LangChain streaming
            async for chunk in self.chain.astream(chain_input):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
            
            # Send completion signal
            is_complete = self.is_conversation_complete(full_response)
            yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response, 'is_complete': is_complete})}\n\n"

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            yield f"data: {json.dumps({'error': 'I apologize, but I am having trouble processing your request right now. Please try again.'})}\n\n"

    def reset_conversation(self) -> List[Dict[str, str]]:
        """Reset conversation to initial state."""
        initial_message = "Hello! I'm here to help you find the perfect water heater type for your home. To give you the best recommendations, I'd like to understand your situation better.\n\nWhat's your main reason for needing a new water heater type? Are you replacing a broken unit, upgrading for better efficiency, building a new home, or something else?"
        
        return [
            {"role": "system", "content": "System initialized"},
            {"role": "assistant", "content": initial_message}
        ]

    def is_conversation_complete(self, response: str) -> bool:
        """Check if the conversation has reached the recommendation stage."""
        completion_indicators = [
            "🏆 personalized water heater type recommendations",
            "ranked recommendations - all 6 systems",
            "annual operating cost:",
            "why #1 for you:",
            "lowest ranked"
        ]
        return any(indicator in response.lower() for indicator in completion_indicators)
    
    def has_sufficient_info(self, messages: List[Dict[str, str]]) -> bool:
        """Determine if enough information has been gathered for recommendations."""
        user_messages = [msg['content'].lower() for msg in messages if msg['role'] == 'user']
        
        essential_info = {
            'situation': ['replace', 'broken', 'new', 'build', 'upgrade', 'old', 'install'],
            'infrastructure': ['gas', 'electric', 'line', 'available', 'connection', 'space'],
            'priorities': ['cost', 'save', 'efficient', 'reliable', 'environment', 'important', 'matter', 'budget', 'cheap', 'expensive'],
            'household': ['family', 'people', 'person', 'use', 'shower', 'bath', 'morning', 'kids']
        }
        
        covered_categories = 0
        priority_mentioned = False
        
        for category, keywords in essential_info.items():
            if any(keyword in ' '.join(user_messages) for keyword in keywords):
                covered_categories += 1
                if category == 'priorities':
                    priority_mentioned = True
        
        return covered_categories >= 3 and priority_mentioned
    
    async def get_response_with_callback(self, messages: List[Dict[str, str]]) -> str:
        """Get a non-streaming response with token usage tracking."""
        try:
            # Convert messages to LangChain format
            chat_history = self._messages_to_langchain_format(messages[:-1])
            current_input = messages[-1]["content"] if messages else ""
            
            chain_input = {
                "input": current_input,
                "chat_history": chat_history
            }
            
            with get_openai_callback() as cb:
                response = await self.chain.ainvoke(chain_input)
                logger.info(f"Token usage: {cb.total_tokens} tokens, Cost: ${cb.total_cost}")
                
            return response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, but I am having trouble processing your request right now. Please try again."

# Initialize the systems with better error handling
try:
    recommendation_system = WaterHeaterRecommendationSystem()
    logger.info("Recommendation system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommendation system: {e}")
    recommendation_system = None

try:
    rag_system = EnhancedRAGFolderStructureDB("Knowledge_Base")
    rag_system.create_database_structure()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

# Global conversation state (in production, use a proper session store)
conversation_sessions = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/start")
async def start_conversation():
    """Start a new conversation."""
    if not recommendation_system:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    session_id = "default"  # In production, generate unique session IDs
    messages = recommendation_system.reset_conversation()
    
    conversation_sessions[session_id] = ConversationState(
        messages=messages,
        turn_count=0
    )
    
    initial_message = messages[-1]['content']  # Get the assistant's initial message
    
    return {
        "session_id": session_id,
        "message": initial_message,
        "is_complete": False
    }

def should_use_rag(message: str, state: ConversationState) -> bool:
    """Determine if we should use RAG system instead of recommendation system."""
    technical_keywords = [
        'how does', 'what is', 'explain', 'tell me about', 'information about',
        'how to', 'what are', 'describe', 'details about', 'more info',
        'how much', 'cost of', 'price of', 'efficiency of', 'lifespan of'
    ]
    
    water_heater_terms = [
        'heat pump', 'tankless', 'solar', 'electric', 'gas', 'installation',
        'maintenance', 'repair', 'efficiency', 'energy', 'temperature',
        'gallon', 'btu', 'warranty', 'brands', 'models', 'thermostat'
    ]
    
    message_lower = message.lower()
    
    # Always use RAG after recommendations are given
    if state.recommendations_given:
        return True
    
    # Use RAG for technical questions during consultation
    has_technical_keyword = any(keyword in message_lower for keyword in technical_keywords)
    has_water_heater_term = any(term in message_lower for term in water_heater_terms)
    
    return has_technical_keyword and has_water_heater_term


async def generate_rag_response(message: str, state: ConversationState):
    """Generate RAG response with proper error handling."""
    try:
        if not rag_system:
            yield f"data: {json.dumps({'content': 'Technical information system is not available right now. Please try again later.', 'done': True})}\n\n"
            return
        
        full_response = ""
        chunk_count = 0
        
        # Use the RAG system
        async for chunk in rag_system.get_full_file_response(message):
            chunk_content = str(chunk) if chunk else ""
            if chunk_content:
                full_response += chunk_content
                chunk_count += 1
                yield f"data: {json.dumps({'content': chunk_content, 'done': False})}\n\n"
        
        # Handle case where no response was generated
        if not full_response or chunk_count == 0:
            fallback_response = "I couldn't find specific information about that topic in my knowledge base. Could you rephrase your question or ask about a different aspect of water heaters?"
            full_response = fallback_response
            yield f"data: {json.dumps({'content': fallback_response, 'done': False})}\n\n"
        
        # Store the conversation
        state.messages.append({"role": "user", "content": message})
        state.messages.append({"role": "assistant", "content": full_response})
        
        # Send completion signal
        yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response, 'is_complete': True, 'using_rag': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in RAG response generation: {e}")
        error_response = "I'm having trouble accessing the technical information right now. Please try rephrasing your question."
        yield f"data: {json.dumps({'content': error_response, 'done': True, 'error': True})}\n\n"

@app.post("/chat/message")
async def send_message(chat_message: ChatMessage):
    """Send a message and get streaming response."""
    if not recommendation_system:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    session_id = "default"
    
    if session_id not in conversation_sessions:
        await start_conversation()
    
    state = conversation_sessions[session_id]
    
    # Handle special commands
    if chat_message.message.lower() in ['restart', 'start over', 'reset']:
        messages = recommendation_system.reset_conversation()
        conversation_sessions[session_id] = ConversationState(
            messages=messages,
            turn_count=0
        )
        return {
            "message": messages[-1]['content'],
            "is_complete": False
        }
    
    # Check if we should use RAG system
    if should_use_rag(chat_message.message, state):
        logger.info(f"Using RAG for question: {chat_message.message}")
        return StreamingResponse(
            generate_rag_response(chat_message.message, state),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    # Use recommendation system for consultation flow
    logger.info(f"Using recommendation system for: {chat_message.message}")
    
    # Force recommendation if requested
    if chat_message.message.lower() in ['recommend', 'give recommendations', 'show options']:
        force_recommendation_msg = "Based on the information provided so far, please provide your detailed recommendations using the specified format."
        state.messages.append({"role": "user", "content": force_recommendation_msg})
    else:
        state.messages.append({"role": "user", "content": chat_message.message})
        state.turn_count += 1
    
    # Check if we should trigger recommendations
    if not state.recommendations_given and state.turn_count >= 4 and recommendation_system.has_sufficient_info(state.messages):
        recommendation_trigger = """
        The user has provided sufficient information about their situation, priorities, and constraints. 
        Now provide comprehensive recommendations for all 6 water heater type systems using the specified format.
        Rank them based on the user's specific situation and priorities discussed.
        """
        state.messages.append({"role": "system", "content": recommendation_trigger})
    
    # Return streaming response from recommendation system
    async def generate_response():
        full_response = ""
        async for chunk in recommendation_system.get_ai_response_stream(state.messages):
            yield chunk
            # Extract full response for checking completion
            if '"done": true' in chunk:
                try:
                    data = json.loads(chunk.split("data: ")[1])
                    if 'full_response' in data:
                        full_response = data['full_response']
                        state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Check if conversation is complete
                        is_complete = recommendation_system.is_conversation_complete(full_response)
                        if is_complete:
                            state.recommendations_given = True
                        final_data = {
                            'content': '',
                            'done': True,
                            'full_response': full_response,
                            'is_complete': is_complete,
                            'using_rag': False
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"
                except:
                    pass
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/chat/status")
async def get_chat_status():
    """Get current chat status."""
    session_id = "default"
    if session_id in conversation_sessions:
        state = conversation_sessions[session_id]
        last_message = state.messages[-1] if state.messages else None
        is_complete = False
        if last_message and last_message['role'] == 'assistant':
            is_complete = recommendation_system.is_conversation_complete(last_message['content'])
        
        return {
            "turn_count": state.turn_count,
            "message_count": len(state.messages),
            "is_complete": is_complete
        }
    return {"error": "No active session"}

@app.post("/chat/non-streaming")
async def send_message_non_streaming(chat_message: ChatMessage):
    """Send a message and get non-streaming response (useful for testing)."""
    if not recommendation_system:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    session_id = "default"
    
    if session_id not in conversation_sessions:
        await start_conversation()
    
    state = conversation_sessions[session_id]
    state.messages.append({"role": "user", "content": chat_message.message})
    state.turn_count += 1
    
    response = await recommendation_system.get_response_with_callback(state.messages)
    state.messages.append({"role": "assistant", "content": response})
    
    is_complete = recommendation_system.is_conversation_complete(response)
    if is_complete:
        state.recommendations_given = True
    
    return {
        "message": response,
        "is_complete": is_complete,
        "turn_count": state.turn_count
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)