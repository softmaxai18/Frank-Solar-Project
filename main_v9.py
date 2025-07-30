#===========================================================================================================================================
#                       IMPORTS & SETUP                      
#===========================================================================================================================================

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Annotated, AsyncGenerator
from typing_extensions import TypedDict
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from colorama import Fore, Style, Back
from datetime import datetime

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

#===========================================================================================================================================
#                       CONSTANTS & CONFIG                   
#===========================================================================================================================================
DEFAULT_SESSION_ID = "default"

DATA_FILES = {
    'summary_table': 'summary_table.json',
}
ALTERNATE_PATHS = [
    Path('All_Context'),
    Path('data'),
    Path('..')
]

#===========================================================================================================================================
#                       GRAPH STATE                          
#===========================================================================================================================================
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    recommendations_given: bool
    using_rag: bool
    last_rag_query: Optional[str]
    last_ai_question: Optional[str]
    classification: Optional[Dict[str, Any]]
    full_response: str
    is_complete: bool
    error: Optional[str]
    decision: Optional[str]
    chart_data: Optional[Dict[str, Any]]

#===========================================================================================================================================
#                       DATA MODELS                          
#===========================================================================================================================================

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)

class ClassificationResult(BaseModel):
    type: str = Field(..., pattern="^(continue_conversation|need_rag_help)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

#===========================================================================================================================================
#                       APPLICATION STATE                    
#===========================================================================================================================================
class AppState:
    """Global application state"""
    def __init__(self):
        self.recommendation_system: Optional['WaterHeaterSystemComponents'] = None
        self.rag_system: Optional['EnhancedRAGFolderStructureDB'] = None
        self.conversation_sessions: Dict[str, GraphState] = {}

app_state = AppState()


#===========================================================================================================================================
#                       DATA LOADING                         
#===========================================================================================================================================
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

#===========================================================================================================================================
#                   SYSTEM COMPONENTS                        
#===========================================================================================================================================
class WaterHeaterSystemComponents:
    """Centralized system components for the graph"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self._load_data()
        self._initialize_components()
        self._initialize_rag()
    
    def _initialize_components(self):
        """Initialize LangChain components"""
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o-mini",
            streaming=True,
            stream_usage=True
        )
        
        self.classifier_llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=100
        )
        
        self.output_parser = StrOutputParser()
        self._create_prompts()

    def _load_data(self):
        """Load all required data files with error handling"""
        try:
            self.summary_table = DataLoader.load_file(DATA_FILES['summary_table'])
            logger.info("Successfully loaded all data files")
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            self._set_fallback_data()
    
    def _set_fallback_data(self):
        """Set fallback data for demo purposes"""
        self.summary_table = {
            "Electric Tank": {"annual_cost": 618.92, "co2": 0.90, "reliability": 4.17},
            "Heat Pump": {"annual_cost": 517.48, "co2": 0.48, "reliability": 1.67},
            "Electric Tankless": {"annual_cost": 670.82, "co2": 0.91, "reliability": 2.50},
            "Active Solar": {"annual_cost": 636.99, "co2": 0.32, "reliability": 1.00},
            "Natural Gas Tankless": {"annual_cost": 308.86, "co2": 0.54, "reliability": 3.33},
            "Natural Gas Tank": {"annual_cost": 336.97, "co2": 0.87, "reliability": 5.00}
        }
    
    def _create_prompts(self):
        """Create the prompt templates"""
        system_prompt_text = f"""
            As a friendly and knowledgeable water heater consultant, your goal is to systematically gather essential information to provide the best recommendation.

            Your personality:
            - Warm, approachable, and conversational - like talking to a trusted neighbor
            - Patient and understanding - you know this can be overwhelming for homeowners
            - Genuinely enthusiastic about helping people find the right fit
            - Thorough but not pushy - you gather information naturally through conversation

            CRITICAL CONVERSATION RULES:
            - Ask EXACTLY one question per response - never combine multiple topics or questions
            - Wait for their complete answer before asking about anything else
            - If you need clarification, focus only on the most important missing piece
            - Never use connecting words like 'and', 'also', 'what about', 'speaking of which'
            - Keep responses to maximum 2 sentences during information gathering
            - Don't repeat the same question - adapt based on their responses

            SYSTEMATIC INFORMATION GATHERING:
            You must collect information in this priority order (ask naturally, not like a checklist):

            1. **SITUATION CONTEXT** - Why they need a water heater
            - Replacement, new installation, upgrade
            - Current system age, type and issues

            2. **UTILITY AVAILABILITY** - What energy sources they have
            - Gas line availability
            - Electrical capacity
            - Both available or limitations

            3. **HOUSEHOLD DETAILS** - Who uses hot water
            - Number of people in household

            4. **USAGE PATTERNS** - How they use hot water
            - Simultaneous usage (multiple showers, appliances)
            - Peak usage times (morning rush, evening)
            - Heavy usage activities (large baths, frequent laundry)

            5. **SPACE & INSTALLATION** - Physical constraints
            - Available space (indoor/outdoor, size limitations)
            - Current location preferences
            - Installation accessibility

            6. **PRIORITIES & BUDGET** - What matters most
            - Speed of heating, energy efficiency, reliability
            - Budget considerations or cost sensitivity
            - Environmental concerns

            7. **PAST EXPERIENCE** - Previous knowledge (types only, not brands)
            - Previous water heater types used
            - Liked or disliked features
            - Maintenance experiences

            CONVERSATION APPROACH:
            - Start with understanding their need
            - Ask follow-up questions that show you're listening and care about details
            - Acknowledge their responses and build on what they share
            - Keep the tone friendly and conversational, not like a formal questionnaire
            - Offer helpful context and insights as you learn about their situation

            THINGS TO AVOID:
            - Never ask about specific brands or manufacturers
            - Don't ask about contractors or installation companies
            - Avoid technical jargon unless necessary
            - Don't ask compound questions or multiple topics at once
            - Don't repeat questions they've already answered

            WHEN TO CONTINUE GATHERING INFO:
            - You need clear answers to at least 5 of the 7 information categories above
            - Keep asking until you understand their situation well enough for a solid recommendation
            - If they give short answers, ask gentle follow-up questions to get more detail

            Remember: You're having a natural conversation to understand their specific needs.
            """
        
        self.system_prompt = system_prompt_text
    
    def _initialize_rag(self):
        """Initialize RAG system"""
        try:
            self.rag_system = EnhancedRAGFolderStructureDB("Knowledge_Base")
            self.rag_system.create_database_structure()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_system = None

# Global system components
system_components = WaterHeaterSystemComponents()

#===========================================================================================================================================
#                       GRAPH NODES                          
#===========================================================================================================================================

async def classify_user_input(state: GraphState) -> GraphState:
    """Node: Classify user input as question or answer"""
    logger.info("Node: classify_user_input")
    log_state(state, "classify_user_input", "IN")
    
    try:
        user_message = state["messages"][-1].content
        conversation_context = state["messages"]
        
        # Get last AI message for context
        conversation_history = []
        for msg in reversed(conversation_context[:-1]):
            if isinstance(msg, (AIMessage, HumanMessage)):
                role = "AI" if isinstance(msg, AIMessage) else "User"
                conversation_history.append(f"{role}: {msg.content.strip()}")
            if len(conversation_history) >= 2:
                break
        
        classification_prompt = f"""
        You are classifying the user's latest message to determine the next action.

        Recent conversation history:
        {conversation_history}

        Latest User Message:
        "{user_message}"

        Decide between:
        - "continue_conversation": if the user is continuing the current structured conversation or answering your previous question
        - "need_rag_help": if the user is asking a NEW technical or fact-based question that might require accessing external documents (RAG)

        Reply ONLY in this JSON format:
        {{
            "type": "continue_conversation" or "need_rag_help",
            "confidence": float (range: 0.0 to 1.0),
            "reasoning": "Brief justification of your classification"
        }}
        """.strip()
        
        response = await system_components.classifier_llm.ainvoke([HumanMessage(content=classification_prompt)])
        
        # Extract JSON from response
        match = re.search(r'\{[\s\S]*?\}', response.content)
        try:
            result_dict = json.loads(match.group(0))
            classification = ClassificationResult(**result_dict)
        except (json.JSONDecodeError, ValueError):
            classification = ClassificationResult(
                type="continue_conversation", confidence=0.5, reasoning="Failed to parse JSON"
            )

        result= {
            **state,
            "classification": classification.model_dump()
        }
        log_state(result, "classify_user_input", "OUT")

        return result

    except Exception as e:
        logger.error(f"Error in classify_user_input: {e}")
        error_state= {
            **state,
            "classification": {"type": "answer", "confidence": 0.5, "reasoning": "Classification failed"},
            "error": str(e)
        }
        log_state(error_state, "classify_user_input", "ERROR")
        return error_state


async def generate_rag_response(state: GraphState) -> Dict[str, Any]:
    """Node: Generate RAG response for technical questions"""
    logger.info("Node: generate_rag_response")
    log_state(state, "generate_rag_response", "IN")
    
    try:
        user_message = state["messages"][-1].content
        
        if not system_components.rag_system:
            return {
                **state,
                "error": "RAG system not initialized",
                "full_response": "I could not access technical information."
            }
        
        # Get relevant file content
        file_content = system_components.rag_system.get_most_relevant_file_content(user_message)
        if not file_content:
            return {
                **state,
                "full_response": "I could not find relevant information in my knowledge base. Please try rephrasing your question.",
                "using_rag": False
            }
        
        # Get last AI message for context continuation
        last_ai_message = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_message = msg.content
                break
        
        file_path = file_content['file_path']
        content = file_content['content']
        similarity = file_content['similarity']
        category = file_content['metadata'].get('category', 'Unknown')
        
        # Create RAG response
        system_prompt = f"""You are an expert water heater consultant.
        You are answering the user's latest question using a complete technical document. After answering, continue the conversation from where it left off previously.

        Guidelines:
        - First, base your answer entirely on the provided document
        - Be thorough and organized
        - Include specific facts, numbers, and technical reasoning
        - Clearly state if something is not covered
        - Then, after completing your answer, continue the conversation by following up on the last AI message

        Previous AI message: "{last_ai_message.strip()}"
        """
                
        user_prompt = f"""Complete Document Content:
        Source: {Path(file_path).name} (Category: {category}, Relevance: {similarity:.2f})

        {content}

        User Question: {user_message}

        1. Please answer this question using the content above.
        2. Then, continue the conversation using this last AI question: "{last_ai_message.strip()}" naturally as if you are following up from the previous assistant message.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Get full response (no streaming here)
        full_response = await system_components.llm.ainvoke(messages)
        full_response = full_response.content if hasattr(full_response, 'content') else str(full_response)
        
        result = {
            **state,
            "full_response": full_response,
            "using_rag": True,
            "last_rag_query": user_message,
            "last_ai_question": last_ai_message  # Preserve the last AI question
        }
        
        log_state(result, "generate_rag_response", "OUT")
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_rag_response: {e}")
        error_state = {
            **state,
            "error": str(e),
            "full_response": "I had trouble accessing technical information."
        }
        log_state(error_state, "generate_rag_response", "ERROR")
        return error_state
    
def add_to_conversation(state: GraphState) -> GraphState:
    """Node: Add user message to conversation history and increment turn count"""
    logger.info("Node: add_to_conversation")
    log_state(state, "add_to_conversation", "IN")
    
    result = {
        **state,
        "turn_count": state.get("turn_count", 0) + 1
    }
    
    log_state(result, "add_to_conversation", "OUT")
    return result

def check_trigger_conditions(state: GraphState) -> GraphState:
    """Node: Check if conditions are met for recommendations with stricter criteria"""
    logger.info("Node: check_trigger_conditions")
    log_state(state, "check_trigger_conditions", "IN")
    
    # Check if already given recommendations
    if state.get("recommendations_given", False):
        result = {**state, "decision": "continue_question"}
        log_state(result, "check_trigger_conditions", "OUT")
        return result

    turn_count = state.get("turn_count", 0)
    messages = state.get("messages", [])
    
    # More strict requirements: at least 8 turns AND sufficient info
    if turn_count >= 9 and has_sufficient_info(messages):
        logger.info(f"Sufficient info gathered after {turn_count} turns")
        result = {**state, "decision": "recommend_types"}
        log_state(result, "check_trigger_conditions", "OUT")
        return result
    
    # More specific force recommendation keywords
    last_message = state["messages"][-1].content.lower()
    force_keywords = [
        'recommend now', 'give me recommendations', 'show me options', 'ready to compare', 
        'what do you recommend', 'help me decide now', 'ready for suggestions',
        'show recommendations', 'give recommendations', 'i want recommendations'
    ]
    
    if any(keyword in last_message for keyword in force_keywords):
        logger.info("Force recommendation triggered by specific keywords")
        result = {**state, "decision": "recommend_types"}
        log_state(result, "check_trigger_conditions", "OUT")
        return result
    
    result = {**state, "decision": "continue_question"}
    log_state(result, "check_trigger_conditions", "OUT")
    return result

async def generate_recommendation(state: GraphState) -> GraphState:
    """Node: Generate structured water heater recommendations"""
    logger.info("Node: generate_recommendation")
    log_state(state, "generate_recommendation", "IN")

    try:
        # Add recommendation trigger to conversation
        system_prompt = """
                The user has provided sufficient information about their situation, priorities, and constraints. 
                Now provide comprehensive recommendations in JSON for all 6 water heater type systems.

                KEY SYSTEM DATA:
                Utilize this information to form your recommendations.

                | System Type           | Annual Cost | Affordability Rank| Fuel Type    | Fuel Supply | Abundance Rank | CO2 Emissions | Environmental Rank | Complexity | Reliability Rank |
                |-----------------------|-------------|-------------------|--------------|-------------|----------------|---------------|-------------------|------------|------------------|
                | Electric Tank         | $618.92/yr  | 1.57              | Electricity  | 88.48 yrs   | 4.47           | 0.90 mt/yr    | 1.07              | 5          | 4.17             |
                | Electric Tankless     | $670.82/yr  | 1.00              | Electricity  | 88.48 yrs   | 4.47           | 0.91 mt/yr    | 1.00              | 3          | 2.50             |
                | Heat Pump             | $517.48/yr  | 2.69              | Electricity  | 88.48 yrs   | 4.47           | 0.48 mt/yr    | 3.92              | 2          | 1.67             |
                | Natural Gas Tank      | $336.97/yr  | 4.69              | Natural Gas  | 86.12 yrs   | 4.36           | 0.87 mt/yr    | 1.27              | 6          | 5.00             |
                | Natural Gas Tankless  | $308.86/yr  | 5.00              | Natural Gas  | 86.12 yrs   | 4.36           | 0.54 mt/yr    | 3.51              | 4          | 3.33             |
                | Active Solar          | $636.99/yr  | 1.37              | Solar Energy | 100.00 yrs  | 5.00           | 0.32 mt/yr    | 5.00              | 1          | 1.00             |

                Ranking Scale: Higher numbers = Better performance (except for Annual Cost and CO2 Emissions where lower is better)
                For recommendations, consider the following factors:

                FUEL TYPE PRIORITY HIERARCHY:
                1. Evaluate by Fuel Type & Availability:
                - Electricity (Electric Tank, Electric Tankless, Heat Pump)
                - Natural Gas (Natural Gas Tank, Natural Gas Tankless)
                - Solar (Active Solar)
                - Local fuel availability
                - Fuel cost and long-term pricing trends

                2. Additional Considerations:
                - Installation complexity
                - Infrastructure constraints
                - Environmental impact prioritization
                - Reliability and maintenance implications

                Important:
                - Ensure the JSON structure is strictly adhered to.
                - Infer ranks based on the data provided without copying values directly.
                - Use User's context to inform recommendations.
                - Base decision-making on fuel type suitability and system performance in context.
                - Validate the JSON format before submission.
                - Convert all numerical values to appropriate ranks based on the provided data.

                The response MUST adhere to the following JSON format without any extra text:

                {
                    "categories": [
                        "Affordability Rank",
                        "Annual Cost",
                        "Abundance Rank",
                        "Environmental Rank",
                        "CO₂ Emissions",
                        "Reliability Rank",
                        "Complexity"
                    ],
                    "heaters": {
                        "Electric Tank":        [?, ?, ?, ?, ?, ?, ?],
                        "Electric Tankless":    [?, ?, ?, ?, ?, ?, ?],
                        "Heat Pump":            [?, ?, ?, ?, ?, ?, ?],
                        "Natural Gas Tank":     [?, ?, ?, ?, ?, ?, ?],
                        "Natural Gas Tankless": [?, ?, ?, ?, ?, ?, ?],
                        "Active Solar":         [?, ?, ?, ?, ?, ?, ?]
                    }
                }
            """
        
        # Get conversation history
        chat_history = messages_to_langchain_format(state["messages"][:-1])
        current_input = state["messages"][-1].content if state["messages"] else ""
        
        messages = [
            SystemMessage(content=system_prompt.strip()),
            *chat_history,
            HumanMessage(content=current_input.strip())
        ]
        
        # Get response
        full_response = await system_components.llm.ainvoke(messages)
        response_content = full_response.content if hasattr(full_response, 'content') else str(full_response)
        logger.info(f"LLM Response: {response_content}...")

        try:
            start = response_content.find('{')
            end = response_content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_text = response_content[start:end+1]
                chart_data = json.loads(json_text)
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Original response: {response_content}")
            
            # Fallback to default data if parsing fails
            chart_data = {
                "categories": [
                    "Affordability Rank",
                    "Annual Cost",
                    "Fuel Supply",
                    "Abundance Rank",
                    "Environmental Rank",
                    "CO₂ Emissions",
                    "Reliability Rank",
                    "Complexity"
                ],
                "heaters": {
                    "Electric Tank": [0,0,0,0,0,0,0,0],
                    "Electric Tankless": [0,0,0,0,0,0,0,0],
                    "Heat Pump": [0,0,0,0,0,0,0,0],
                    "Natural Gas Tank": [0,0,0,0,0,0,0,0],
                    "Natural Gas Tankless": [0,0,0,0,0,0,0,0],
                    "Active Solar": [0,0,0,0,0,0,0,0]
                }
            }
        
        
        result = {
            **state,
            "full_response":  json.dumps(chart_data),
            "is_complete": True,
            "using_rag": False,
            "recommendations_given": True,
            "chart_data": chart_data
        }
        
        log_state(result, "generate_recommendation", "OUT")
        return result

    except Exception as e:
        logger.error(f"Error in generate_recommendation: {e}")
        error_state = {
            **state,
            "error": str(e),
            "full_response": "I had trouble generating recommendations.",
            "is_complete": False
        }
        log_state(error_state, "generate_recommendation", "ERROR")
        return error_state

async def continue_conversation(state: GraphState) -> Dict[str, Any]:
    """Node: Continue normal conversation flow"""
    logger.info("Node: continue_conversation")
    log_state(state, "continue_conversation", "IN")

    try:
        # Get conversation history
        chat_history = messages_to_langchain_format(state["messages"][:-1])
        current_input = state["messages"][-1].content if state["messages"] else ""
        
        messages = [
            SystemMessage(content=system_components.system_prompt.strip()),
            *chat_history,
            HumanMessage(content=current_input.strip())
        ]

        # Send messages to the LLM
        full_response = await system_components.llm.ainvoke(messages)
        full_response = full_response.content if hasattr(full_response, 'content') else str(full_response)
        is_complete = is_conversation_complete(full_response)
        
        result = {
            **state,
            "full_response": full_response,
            "is_complete": is_complete,
            "using_rag": False
        }
        
        log_state(result, "continue_conversation", "OUT")
        return result
        
    except Exception as e:
        logger.error(f"Error in continue_conversation: {e}")
        error_state = {
            **state,
            "error": str(e),
            "full_response": "I apologize, but I encountered an error processing your message.",
            "is_complete": False
        }
        log_state(error_state, "continue_conversation", "ERROR")
        return error_state

def save_response(state: GraphState) -> GraphState:
    """Node: Save assistant response to conversation history"""
    logger.info("Node: save_response")
    log_state(state, "save_response", "IN")

    try:
        # Add the AI response to messages
        ai_message = AIMessage(content=state["full_response"])
        
        # Save to file (optional)
        save_conversation_to_file(state["messages"], state["full_response"])
        
        result = {
            **state,
            "messages": state["messages"] + [ai_message],
            "last_ai_question": state["full_response"] if not state.get("using_rag", False) else state.get("last_ai_question", "")
        }
        
        log_state(result, "save_response", "OUT")
        return result
        
    except Exception as e:
        logger.error(f"Error in save_response: {e}")
        result = {
            **state,
            "messages": state["messages"] + [AIMessage(content="I encountered an error processing your request.")],
            "error": str(e)
        }
        
        log_state(result, "save_response", "OUT")
        return result

#===========================================================================================================================================
#                       UTILITY FUNCTIONS                    
#===========================================================================================================================================

def messages_to_langchain_format(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert messages to LangChain format"""
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage, SystemMessage))]

def has_sufficient_info(messages: List[BaseMessage]) -> bool:
    """Enhanced check for sufficient information with stricter requirements"""
    user_messages = [msg.content.lower() for msg in messages if isinstance(msg, HumanMessage)]
    combined_text = ' '.join(user_messages)
    
    # Comprehensive information categories (need 5 out of 7)
    essential_info = {
        'situation': ['replace', 'broken', 'new', 'build', 'upgrade', 'old', 'install', 'need', 'want', 'emergency'],
        'infrastructure': ['gas', 'electric', 'line', 'available', 'connection', 'both', 'electricity', 'natural gas', '220v', 'voltage'],
        'household': ['family', 'people', 'person', 'use', 'kids', 'adults', 'children', 'teenagers', 'elderly'],
        'usage_patterns': ['shower', 'bath', 'morning', 'evening', 'peak', 'usage', 'demand', 'simultaneous', 'laundry', 'dishwasher'],
        'space': ['space', 'room', 'enough', 'big', 'small', 'basement', 'garage', 'outdoor', 'indoor', 'limitation', 'constraint', 'location'],
        'priorities': ['budget', 'cost', 'efficient', 'fast', 'heating', 'reliable', 'environmental', 'save', 'money', 'speed', 'quick'],
        'experience': ['previous', 'before', 'last', 'experience', 'liked', 'disliked', 'problem', 'maintenance']
    }
    
    covered_categories = 0
    for category, keywords in essential_info.items():
        if any(keyword in combined_text for keyword in keywords):
            covered_categories += 1
    
    # Need at least 5 out of 7 categories covered
    return covered_categories >= 6

def is_conversation_complete(response: str) -> bool:
    """Check if the conversation has reached the recommendation stage"""
    completion_indicators = [
        "🏆 personalized water heater type recommendations",
        "ranked recommendations - all 6 systems",
        "🥇 ranked recommendations",
        "annual operating cost:",
        "why #1 for you:",
        "lowest ranked",
        "detailed analysis for your top choice"
    ]
    return any(indicator in response.lower() for indicator in completion_indicators)

def log_state(state: GraphState, node_name: str, direction: str = "IN"):
    """Enhanced state logging"""
    
    color_map = {
        "IN": Fore.CYAN,
        "OUT": Fore.GREEN,
        "ERROR": Fore.RED
    }
    color = color_map.get(direction, Fore.WHITE)
    reset = Style.RESET_ALL

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    logger.info(f"{color}==== {timestamp} ==== {direction} {node_name.upper()} ===={reset}")
    
    # Clean and concise message display
    simplified_messages = [
        f"{msg.__class__.__name__}: {msg.content.strip()[:50]}"
        for msg in state.get("messages", [])
    ]

    # Essential fields to always log
    essential_fields = {
        "messages": simplified_messages,
        "turn_count": state.get("turn_count", 0),
        "recommendations_given": state.get("recommendations_given", False),
        "using_rag": state.get("using_rag", False),
        "last_rag_query": state.get("last_rag_query", ""),
        "last_ai_question":(state['last_ai_question'][:50] + "...") if state.get("full_response", "") else "None",
        "classification": state.get("classification", {}),
        "full_response": (state['full_response'][:50] + "...") if state.get("full_response", "") else "None",
        "is_complete": state.get("is_complete", False),
        "error": state.get("error"),
        "message_count": len(state.get("messages", [])),
        "decision": state.get("decision", ""),
        "chart_data": state.get("chart_data", {})
    }
    
    for field, value in essential_fields.items():
        print(f"{Fore.YELLOW}{field:>20}{reset}: {str(value).strip()}")
    
    logger.info(f"{color}==== {timestamp} ==== END {direction} ================{reset}\n")

def save_conversation_to_file(messages: List[BaseMessage], ai_response: str, filename: str = "conversation_log.txt"):
    """Save conversation to a file"""
    try:
        with open(filename, "a", encoding="utf-8") as f:
            if messages:
                last_user_msg = messages[-1]
                if isinstance(last_user_msg, HumanMessage):
                    f.write(f'UserMessage="{last_user_msg.content}"\n')
            f.write(f'AIMessage="{ai_response}"\n\n')
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

#===========================================================================================================================================
#                       GRAPH CONSTRUCTION                   
#===========================================================================================================================================

def create_water_heater_graph():
    """Create and return the LangGraph workflow"""
    
    # Create the graph
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("classify", classify_user_input)
    builder.add_node("rag_query", generate_rag_response)
    builder.add_node("add_to_conversation", add_to_conversation)
    builder.add_node("check_recommend", check_trigger_conditions)
    builder.add_node("recommend", generate_recommendation)
    builder.add_node("continue", continue_conversation)
    builder.add_node("save_response", save_response)
    
    # Set entry point
    builder.set_entry_point("classify")

    # Add conditional edges
    builder.add_conditional_edges(
        "classify", 
        lambda state: state["classification"]["type"] if state.get("classification") else "continue_conversation",
        {
            "need_rag_help": "rag_query",
            "continue_conversation": "add_to_conversation"
        }
    )
    
    # Add regular edges
    builder.add_edge("rag_query", "save_response")
    builder.add_edge("add_to_conversation", "check_recommend")
    
    builder.add_conditional_edges(
        "check_recommend",
        lambda state: state.get("decision", "continue_question"),
        {
            "recommend_types": "recommend",
            "continue_question": "continue"
        }
    )
    
    builder.add_edge("recommend", "save_response")
    builder.add_edge("continue", "save_response")
    builder.add_edge("save_response", END)
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

#===========================================================================================================================================
#                       MAIN INTERFACE                       
#===========================================================================================================================================

class WaterHeaterGraphInterface:
    """Main interface for the Water Heater recommendation system"""
    
    def __init__(self):
        self.graph = create_water_heater_graph()
        self.config = {"configurable": {"thread_id": "default"}}
    
    def start_conversation(self) -> str:
        """Start a new conversation using LLM"""
        app_state.conversation_sessions.clear()
        self.graph = create_water_heater_graph()

        new_thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": new_thread_id}}
        
        system_message = SystemMessage(content=system_components.system_prompt + "\n\nStart the conversation with a warm greeting and ask your first question to understand their situation.")
        response = system_components.llm.invoke([system_message, HumanMessage(content="Begin the conversation")])
        initial_message = response.content
        
        # Initialize state
        initial_state = {
        "messages": [AIMessage(content=initial_message)],
        "turn_count": 0, 
        "recommendations_given": False,
        "using_rag": False,
        "last_rag_query": None,
        "last_ai_question": initial_message,  # Track initial question
        "classification": None,
        "full_response": initial_message,
        "is_complete": False,
        "error": None,
        "decision": None  # Add decision field
    }
        
        app_state.conversation_sessions[new_thread_id] = initial_state
        return initial_state
    
    async def process_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """Process a user message through the graph with streaming output"""
        try:
            session_id = DEFAULT_SESSION_ID
            state = get_conversation_state(session_id)
            # Add the new user message
            state["messages"].append(HumanMessage(content=user_message))
            # state["turn_count"] = state.get("turn_count", 0) + 1

            fallback_sent = False

            async for msg, metadata in self.graph.astream(state, stream_mode="messages", config=self.config):
                if isinstance(msg, AIMessage):
                    if metadata.get("langgraph_node") in ["continue", "rag_query", "recommend"]:
                        yield f"data: {json.dumps({'content': msg.content})}\n\n"
                        fallback_sent = True

            final_state = self.graph.get_state(self.config)
            if final_state and final_state.values:
                app_state.conversation_sessions[session_id].update(final_state.values)

            if not fallback_sent:
                yield f"data: {json.dumps({'content': 'I\'m sorry, I couldn\'t generate a meaningful response. Could you rephrase or ask something else?'})}\n\n"


        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


#===========================================================================================================================================
#                       DEPENDENCIES                         
#===========================================================================================================================================
def get_recommendation_system() -> WaterHeaterGraphInterface:
    """Get recommendation system instance"""
    if not app_state.recommendation_system:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    return app_state.recommendation_system

def get_conversation_state(session_id: str = DEFAULT_SESSION_ID) -> GraphState:
    """Get or create conversation state for session"""
    if session_id not in app_state.conversation_sessions:
        app_state.conversation_sessions[session_id] = {
            "messages": [],
            "turn_count": 0,
            "recommendations_given": False,
            "using_rag": False,
            "last_rag_query": None,
            "last_ai_question": None,
            "classification": None,
            "full_response": "",
            "is_complete": False,
            "error": None,
            "decision": None
        }
    return app_state.conversation_sessions[session_id]


#===========================================================================================================================================
#                       APP LIFECYCLE                        
#===========================================================================================================================================
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting application...")
    
    try:
        app_state.recommendation_system = WaterHeaterGraphInterface()
        logger.info("Recommendation system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation system: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


#===========================================================================================================================================
#                       FASTAPI APP                          
#===========================================================================================================================================
# FastAPI app
app = FastAPI(
    title="Water Heater Type Recommendation System",
    description="AI-powered water heater recommendation system with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Templates
templates = Jinja2Templates(directory="templates")


#===========================================================================================================================================
#                       API ROUTES                           
#===========================================================================================================================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index_4.html", {"request": request})

@app.post("/chat/start")
async def start_conversation(
    recommendation_system: WaterHeaterGraphInterface = Depends(get_recommendation_system)
):
    """Start a new conversation"""
    initial_state = recommendation_system.start_conversation()
    app_state.conversation_sessions[DEFAULT_SESSION_ID] = initial_state
    
    return {
        "session_id": DEFAULT_SESSION_ID,
        "message": initial_state["full_response"],
        "is_complete": False
    }

@app.post("/chat/message")
async def send_message(
    chat_message: ChatMessage,
    recommendation_system: WaterHeaterGraphInterface = Depends(get_recommendation_system)
):
    """Send a message and get streaming response"""
    
    # Handle special commands
    if chat_message.message.lower() in ['restart', 'start over', 'reset']:
        initial_state = recommendation_system.start_conversation()
        app_state.conversation_sessions[DEFAULT_SESSION_ID] = initial_state
        return {"message": initial_state["full_response"], "is_complete": False}
    
    return StreamingResponse(
        recommendation_system.process_message(chat_message.message),
        media_type="text/event-stream"
    )

@app.get("/chat/chart-data")
async def get_chart_data():
    """Return the recommendation chart data"""
    state = get_conversation_state(DEFAULT_SESSION_ID) 
    if not state.get("chart_data"):
        raise HTTPException(status_code=404, detail="Chart data not found")
    
    return state["chart_data"]

@app.post("/chat/click-log")
async def click_log(request: Request):
    data = await request.json()
    print("Bar Clicked:", data)
    return {"status": "success", "received": data}

@app.get("/chat/status")
async def get_chat_status(
    recommendation_system: WaterHeaterGraphInterface = Depends(get_recommendation_system),
    state: GraphState = Depends(get_conversation_state)
):
    """Get current chat status"""
    last_message = state["messages"][-1] if state["messages"] else None
    is_complete = False
    
    if last_message and isinstance(last_message, AIMessage):  # Fixed: use isinstance instead of dict access
        is_complete = is_conversation_complete(last_message.content)
    
    return {
        "turn_count": state.get("turn_count", 0),
        "message_count": len(state.get("messages", [])),
        "is_complete": is_complete,
        "recommendations_given": state.get("recommendations_given", False)
    }

#===========================================================================================================================================
#                       MAIN ENTRY                          
#===========================================================================================================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        port=8000,
        log_level="info"
    )