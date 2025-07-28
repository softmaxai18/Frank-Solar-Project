#============================================================
#                       IMPORTS & SETUP                      
#============================================================
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback   

# Import your RAG system
from RAG_Openai import EnhancedRAGFolderStructureDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


#============================================================
#                       CONSTANTS & CONFIG                   
#============================================================
# Constants
DEFAULT_SESSION_ID = "default"
DATA_FILES = {
    'summary_table': 'summary_table.json',
}
ALTERNATE_PATHS = [
    Path('All_Context'),
    Path('data'),
    Path('..')
]


#============================================================
#                       DATA MODELS                          
#============================================================
# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)

class ChatResponse(BaseModel):
    response: str
    is_complete: bool = False
    recommendations_given: bool = False

class ConversationState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    turn_count: int = 0
    recommendations_given: bool = False
    using_rag: bool = False
    last_rag_query: Optional[str] = None
    last_ai_question: Optional[str] = None

class ClassificationResult(BaseModel):
    type: str = Field(..., pattern="^(question|answer)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


#============================================================
#                       APPLICATION STATE                    
#============================================================
class AppState:
    """Global application state"""
    def __init__(self):
        self.recommendation_system: Optional['WaterHeaterRecommendationSystem'] = None
        self.rag_system: Optional['EnhancedRAGFolderStructureDB'] = None
        self.conversation_sessions: Dict[str, ConversationState] = {}

app_state = AppState()


#============================================================
#                       DATA LOADING                         
#============================================================
class DataLoader:
    """Handles loading and caching of data files"""
    
    @staticmethod
    def load_file(filename: str) -> Any:
        """Load a file with automatic path resolution"""
        file_path = Path(filename)
        
        if not file_path.exists():
            logger.warning(f"File {filename} not found. Trying alternate paths...")
            for alt_path in ALTERNATE_PATHS:
                potential_path = alt_path / filename
                if potential_path.exists():
                    file_path = potential_path
                    break
            else:
                raise FileNotFoundError(f"Could not find {filename} in any expected location")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if filename.endswith('.json'):
                    return json.load(f)
                return f.read()
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise


#============================================================
#                   RECOMMENDATION SYSTEM                   
#============================================================
class WaterHeaterRecommendationSystem:
    """Optimized water heater recommendation system"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self._initialize_components()
        self._load_data()
        self._create_chain()
    
    def _initialize_components(self) -> None:
        """Initialize LangChain components"""
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o-mini",
            streaming=True,
            stream_usage=True
        )
        self.output_parser = StrOutputParser()
        
        # Initialize classifier for input classification
        self.classifier_llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=100
        )
    
    def _load_data(self) -> None:
        """Load all required data files with error handling"""
        try:
            self.summary_table = DataLoader.load_file(DATA_FILES['summary_table'])
            logger.info("Successfully loaded all data files")
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            self._set_fallback_data()
    
    def _set_fallback_data(self) -> None:
        """Set fallback data for demo purposes"""
        self.summary_table = {
            "Electric Tank": {"annual_cost": 618.92, "co2": 0.90, "reliability": 4.17},
            "Heat Pump": {"annual_cost": 517.48, "co2": 0.48, "reliability": 1.67},
            "Electric Tankless": {"annual_cost": 670.82, "co2": 0.91, "reliability": 2.50},
            "Active Solar": {"annual_cost": 636.99, "co2": 0.32, "reliability": 1.00},
            "Natural Gas Tankless": {"annual_cost": 308.86, "co2": 0.54, "reliability": 3.33},
            "Natural Gas Tank": {"annual_cost": 336.97, "co2": 0.87, "reliability": 5.00}
        }
    
    def _create_chain(self) -> None:
        """Create the LangChain processing chain"""
        system_prompt_text = f"""
            You are an expert water heater type consultant with access to comprehensive technical data and cost analysis for 6 different water heating systems. Your role is to conduct a natural, conversational interview to understand the user's specific situation, then provide personalized rankings of all 6 systems based on a structured fuel-type selection framework.

            THESE ARE WATER HEATER SYSTEMS TYPE not WATER HEATER TYPE:
            {json.dumps(self.summary_table, indent=2).replace('{', '{{').replace('}', '}}') if self.summary_table else "No summary data available"}

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
        
        self.system_prompt_template = SystemMessagePromptTemplate.from_template(system_prompt_text)
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt_template,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        self.chain = (
            {
                "chat_history": lambda x: x.get("chat_history", []),
                "input": RunnablePassthrough()
            }
            | self.chat_prompt
            | self.llm
            | self.output_parser
        )
    
    @staticmethod
    def messages_to_langchain_format(messages: List[Dict[str, str]]) -> List:
        """Convert internal message format to LangChain message format"""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages
    
    async def get_ai_response_stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Get streaming response from LangChain"""
        try:
            chat_history = self.messages_to_langchain_format(messages[:-1])
            current_input = messages[-1]["content"] if messages else ""
            
            chain_input = {
                "input": current_input,
                "chat_history": chat_history
            }

            
            full_response = ""
            with get_openai_callback() as cb:
                async for chunk in self.chain.astream(chain_input):
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"

            is_complete = self.is_conversation_complete(full_response)


            logger.info(
                f"Total Tokens: {cb.total_tokens} "
                f"Prompt Tokens: {cb.prompt_tokens} "
                f"Completion Tokens: {cb.completion_tokens} "
                f"Total Cost (USD): ${cb.total_cost}"
            )


            yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response, 'is_complete': is_complete})}\n\n"
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            yield f"data: {json.dumps({'error': 'I apologize, but I am having trouble processing your request right now. Please try again.'})}\n\n"
    
    async def classify_user_input(self, user_message: str, conversation_context: List[Dict[str, str]]) -> ClassificationResult:
        """Classify user input as either a question or an answer"""
        last_ai_message = ""

        for msg in reversed(conversation_context):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # ✅ Check for RAG keywords (from your template) and length
                is_rag_like = "Complete Document Content" in content or "Source:" in content
                if is_rag_like or len(content.split()) > 200:
                    continue  # skip RAG-based long messages
                last_ai_message = content
                break

        classification_prompt = f"""
            TASK:
            Classify the user's input as either a "question" (seeking information) or an "answer" (responding to a previous message), based on the conversation context.

            CONTEXT:
            Previous AI message: "{last_ai_message}"
            User input: "{user_message}"

            CLASSIFICATION GUIDELINES:
            - Classify as "question" if:
                - The user is asking for question, information, clarification, or explanation (e.g., "How does a water heater work?")
                - The message is clearly a new query not logically responding to the previous AI message

            - Classify as "answer" if:
                - The user is directly responding to the previous AI message with information, feedback, or a solution
                - The message is a continuation or completion of the conversation initiated by the AI

            OUTPUT FORMAT:
            Return only a JSON object in the following format:

            ```json
            {{
            "type": "question" or "answer",
            "confidence": float (range: 0.0 to 1.0),
            "reasoning": "Brief justification of your classification decision"
            }}
        """
        
        try:
            response = await self.classifier_llm.ainvoke(classification_prompt.strip())
            response_content = getattr(response, 'content', str(response)).strip()

            # Extract the first JSON-like object from the response
            match = re.search(r'\{[\s\S]*?\}', response_content)
            if match:
                try:
                    result_dict = json.loads(match.group(0))
                    return ClassificationResult(**result_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse classification result: {e}")
            
            return ClassificationResult(
                type="answer",
                confidence=0.5,
                reasoning="Could not parse AI output, defaulting to answer"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error during classification: {e}")
            return ClassificationResult(
                type="answer",
                confidence=0.5,
                reasoning="Classification failed, defaulting to answer"
            )
    
    def reset_conversation(self) -> List[Dict[str, str]]:
        """Reset conversation to initial state"""
        initial_message = ("Hello! I'm here to help you find the perfect water heater type for your home. "
                         "To give you the best recommendations, I'd like to understand your situation better.\n\n"
                         "What's your main reason for needing a new water heater type? Are you replacing a broken unit, "
                         "upgrading for better efficiency, building a new home, or something else?")
        
        return [
            {"role": "system", "content": "System initialized"},
            {"role": "assistant", "content": initial_message}
        ]
    
    @staticmethod
    def is_conversation_complete(response: str) -> bool:
        """Check if the conversation has reached the recommendation stage"""
        completion_indicators = [
            "🏆 personalized water heater type recommendations",
            "ranked recommendations - all 6 systems",
            "annual operating cost:",
            "why #1 for you:",
            "lowest ranked"
        ]
        return any(indicator in response.lower() for indicator in completion_indicators)
    
    @staticmethod
    def has_sufficient_info(messages: List[Dict[str, str]]) -> bool:
        """Determine if enough information has been gathered for recommendations"""
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


#============================================================
#                   CONVERSATION MANAGEMENT                  
#============================================================
class ConversationManager:
    """Manages conversation state and logging"""
    
    @staticmethod
    def save_conversation_to_file(messages: List[Dict[str, str]], filename: str = "conversation_log.txt") -> None:
        """Save conversation to a file"""
        try:
            with open(filename, "a", encoding="utf-8") as f:
                for msg in messages[-2:]:  # Only save the last exchange
                    if msg["role"] == "user":
                        f.write(f'UserMessage="{msg["content"]}"\n')
                    elif msg["role"] == "assistant":
                        f.write(f'AIMessage="{msg["content"]}"\n\n')
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")


#============================================================
#                       RAG HANDLING                        
#============================================================
async def generate_rag_response_direct(
    message: str,
    state: ConversationState,
    conversation_context: List[Dict[str, str]]
) -> AsyncGenerator[str, None]:
    """Direct RAG response using main file LLM and RAG file content, then continues conversation from last AI question"""
    
    # Get the last AI message to continue the conversation after RAG
    last_ai_message = ""
    for msg in reversed(conversation_context):
        if msg.get("role") == "assistant":
            last_ai_message = msg.get("content", "")
            break    

    recommendation_system = app_state.recommendation_system
    try:
        rag_system = app_state.rag_system
        if not rag_system:
            yield f"data: {json.dumps({'content': 'Technical information system is not available right now.', 'done': True})}\n\n"
            return

        file_content = rag_system.get_most_relevant_file_content(message)
        if not file_content:
            error_msg = (
                "I couldn't find a relevant file in my knowledge base to answer your question. "
                "Please try rephrasing your question or ask about specific water heater topics."
            )
            yield f"data: {json.dumps({'content': error_msg, 'done': True, 'using_rag': True})}\n\n"
            return

        file_path = file_content['file_path']
        content = file_content['content']
        similarity = file_content['similarity']
        category = file_content['metadata'].get('category', 'Unknown')

        # Updated system prompt to continue from previous conversation
        system_prompt = f"""You are an expert water heater consultant.
        You are answering the user's latest question using a complete technical document. After answering, continue the conversation from where it left off previously.

        Guidelines:
        - First, base your answer entirely on the provided document
        - Be thorough and organized
        - Include specific facts, numbers, and technical reasoning
        - Clearly state if something is not covered
        - Then, after completing your answer, continue the conversation by following up on the last AI message (context below)

        Previous AI message :
        \"{last_ai_message.strip()}\"
        """

        user_prompt = f"""Complete Document Content:
        Source: {Path(file_path).name} (Category: {category}, Relevance: {similarity:.2f})

        {content}

        User Question: {message}

        1. Please answer this question using the content above.
        2. Then, continue the conversation usign this last ai question:\"{last_ai_message.strip()}\" naturally as if you are following up from the previous assistant message shown above.
        """

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        llm = recommendation_system.llm 
        full_response = ""

        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content'):
                content_chunk = chunk.content
            else:
                content_chunk = str(chunk)

            full_response += content_chunk
            yield f"data: {json.dumps({'content': content_chunk, 'done': False})}\n\n"

        final_data = {
            'content': '',
            'done': True,
            'full_response': full_response,
            'is_complete': False,
            'using_rag': True
        }
        yield f"data: {json.dumps(final_data)}\n\n"

        # Save messages
        state.messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response}
        ])
        ConversationManager.save_conversation_to_file(state.messages)

    except Exception as e:
        logger.error(f"RAG response error: {e}")
        error_response = "I am having trouble accessing technical information right now."
        yield f"data: {json.dumps({'content': error_response, 'done': True, 'error': True})}\n\n"


#============================================================
#                       DEPENDENCIES                         
#============================================================
def get_recommendation_system() -> WaterHeaterRecommendationSystem:
    """Get recommendation system instance"""
    if not app_state.recommendation_system:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    return app_state.recommendation_system

def get_conversation_state(session_id: str = DEFAULT_SESSION_ID) -> ConversationState:
    """Get or create conversation state for session"""
    if session_id not in app_state.conversation_sessions:
        app_state.conversation_sessions[session_id] = ConversationState()
    return app_state.conversation_sessions[session_id]


#============================================================
#                       APP LIFECYCLE                        
#============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting application...")
    
    try:
        app_state.recommendation_system = WaterHeaterRecommendationSystem()
        logger.info("Recommendation system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation system: {e}")
    
    try:
        app_state.rag_system = EnhancedRAGFolderStructureDB("Knowledge_Base")
        app_state.rag_system.create_database_structure()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


#============================================================
#                       FASTAPI APP                          
#============================================================
# FastAPI app
app = FastAPI(
    title="Water Heater Type Recommendation System",
    description="AI-powered water heater recommendation system with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Templates
templates = Jinja2Templates(directory="templates")


#============================================================
#                       API ROUTES                           
#============================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/start")
async def start_conversation(
    recommendation_system: WaterHeaterRecommendationSystem = Depends(get_recommendation_system)
):
    """Start a new conversation"""
    messages = recommendation_system.reset_conversation()
    
    state = ConversationState(messages=messages, turn_count=0)
    app_state.conversation_sessions[DEFAULT_SESSION_ID] = state
    
    return {
        "session_id": DEFAULT_SESSION_ID,
        "message": messages[-1]['content'],
        "is_complete": False
    }

@app.post("/chat/message")
async def send_message(
    chat_message: ChatMessage,
    recommendation_system: WaterHeaterRecommendationSystem = Depends(get_recommendation_system),
    state: ConversationState = Depends(get_conversation_state)
):
    """Send a message and get streaming response with input classification"""
    
    # Handle special commands
    if chat_message.message.lower() in ['restart', 'start over', 'reset']:
        messages = recommendation_system.reset_conversation()
        app_state.conversation_sessions[DEFAULT_SESSION_ID] = ConversationState(
            messages=messages, turn_count=0
        )
        ConversationManager.save_conversation_to_file(messages)
        return {"message": messages[-1]['content'], "is_complete": False}
    
    # Classify user input
    classification = await recommendation_system.classify_user_input(chat_message.message, state.messages)
    logger.info(f"Input classification: {classification}")
    
    if classification.type == "question" and classification.confidence > 0.6:        
        return StreamingResponse(
            generate_rag_response_direct(chat_message.message, state, state.messages),
            media_type="text/event-stream"
        )
    
    else:
        # Use recommendation system for answers
        if chat_message.message.lower() in ['recommend', 'give recommendations', 'show options']:
            force_recommendation_msg = "Based on the information provided so far, please provide your detailed recommendations using the specified format."
            state.messages.append({"role": "user", "content": force_recommendation_msg})
        else:
            state.messages.append({"role": "user", "content": chat_message.message})
            state.turn_count += 1
        
        # Check if we should trigger recommendations
        if (not state.recommendations_given and 
            state.turn_count >= 4 and 
            recommendation_system.has_sufficient_info(state.messages)):
            
            recommendation_trigger = """
            The user has provided sufficient information about their situation, priorities, and constraints. 
            Now provide comprehensive recommendations for all 6 water heater type systems using the specified format.
            Rank them based on the user's specific situation and priorities discussed.
            """
            state.messages.append({"role": "system", "content": recommendation_trigger})
        
        async def generate_response():
            full_response = ""
            async for chunk in recommendation_system.get_ai_response_stream(state.messages):
                yield chunk
                if '"done": true' in chunk:
                    try:
                        data = json.loads(chunk.split("data: ")[1])
                        if 'full_response' in data:
                            full_response = data['full_response']
                            state.messages.append({"role": "assistant", "content": full_response})
                            ConversationManager.save_conversation_to_file(state.messages)
                            
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
async def get_chat_status(
    recommendation_system: WaterHeaterRecommendationSystem = Depends(get_recommendation_system),
    state: ConversationState = Depends(get_conversation_state)
):
    """Get current chat status"""
    last_message = state.messages[-1] if state.messages else None
    is_complete = False
    
    if last_message and last_message['role'] == 'assistant':
        is_complete = recommendation_system.is_conversation_complete(last_message['content'])
    
    return {
        "turn_count": state.turn_count,
        "message_count": len(state.messages),
        "is_complete": is_complete,
        "recommendations_given": state.recommendations_given
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "recommendation_system": app_state.recommendation_system is not None,
        "rag_system": app_state.rag_system is not None
    }


#============================================================
#                       MAIN ENTRY                          
#============================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        port=8000,
        log_level="info"
    )