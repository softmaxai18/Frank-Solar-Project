#============================================================
#                       IMPORTS & SETUP                      
#============================================================

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Annotated, AsyncGenerator
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.callbacks import get_openai_callback

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


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
#                       GRAPH STATE                          
#============================================================
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

#============================================================
#                       DATA MODELS                          
#============================================================

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)

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
        self.recommendation_system: Optional['WaterHeaterSystemComponents'] = None
        self.rag_system: Optional['EnhancedRAGFolderStructureDB'] = None
        self.conversation_sessions: Dict[str, GraphState] = {}

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
#                   SYSTEM COMPONENTS                        
#============================================================
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
        """
        
        self.system_prompt_template = SystemMessagePromptTemplate.from_template(system_prompt_text)
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt_template,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
    
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

#============================================================
#                       GRAPH NODES                          
#============================================================

def classify_user_input(state: GraphState) -> GraphState:
    """Node: Classify user input as question or answer"""
    logger.info("Node: classify_user_input")
    
    try:
        user_message = state["messages"][-1].content
        conversation_context = state["messages"]
        
        # Get last AI message for context
        last_ai_message = ""
        for msg in reversed(conversation_context):
            if isinstance(msg, AIMessage):
                content = msg.content
                # Skip RAG-based long messages
                is_rag_like = "Complete Document Content" in content or "Source:" in content
                if is_rag_like or len(content.split()) > 200:
                    continue
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
                - The user is asking for information, clarification, or explanation
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
        
        response = system_components.classifier_llm.invoke(classification_prompt.strip())
        response_content = response.content.strip()
        
        # Extract JSON from response
        match = re.search(r'\{[\s\S]*?\}', response_content)
        if match:
            try:
                result_dict = json.loads(match.group(0))
                classification = ClassificationResult(**result_dict)
            except (json.JSONDecodeError, ValueError):
                classification = ClassificationResult(
                    type="answer", confidence=0.5, reasoning="Could not parse AI output"
                )
        else:
            classification = ClassificationResult(
                type="answer", confidence=0.5, reasoning="No JSON found in response"
            )
        
        return {
            **state,
            "classification": classification.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error in classify_user_input: {e}")
        return {
            **state,
            "classification": {"type": "answer", "confidence": 0.5, "reasoning": "Classification failed"},
            "error": str(e)
        }


async def generate_rag_response(state: GraphState) -> Dict[str, Any]:
    """Node: Generate RAG response for technical questions"""
    logger.info("Node: generate_rag_response")
    
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
        
        return {
            **state,
            "full_response": full_response,
            "using_rag": True,
            "last_rag_query": user_message
        }
        
    except Exception as e:
        logger.error(f"Error in generate_rag_response: {e}")
        return {
            **state,
            "error": str(e),
            "full_response": "I had trouble accessing technical information."
        }
        
def add_to_conversation(state: GraphState) -> GraphState:
    """Node: Add user message to conversation history and increment turn count"""
    logger.info("Node: add_to_conversation")
    
    return {
        **state,
        "turn_count": state.get("turn_count", 0) + 1
    }

def check_trigger_conditions(state: GraphState) -> GraphState:
    """Node: Check if conditions are met for recommendations"""
    logger.info("Node: check_trigger_conditions")
    
    # Check if already given recommendations
    if state.get("recommendations_given", False):
        return {**state, "decision": "continue_question"}
    
    # Check if sufficient information gathered
    turn_count = state.get("turn_count", 0)
    messages = state.get("messages", [])
    
    # More comprehensive check for sufficient information
    if turn_count >= 4 or has_sufficient_info(messages):
        logger.info(f"Sufficient info gathered after {turn_count} turns")
        return {**state, "decision": "recommend_types"}
    
    # Check for force recommendation keywords
    last_message = state["messages"][-1].content.lower()
    force_keywords = [
        'recommend', 'give recommendations', 'show options', 'compare', 
        'which one', 'best option', 'what do you recommend', 'suggest',
        'help me choose', 'pick one', 'decision', 'choose'
    ]
    
    if any(keyword in last_message for keyword in force_keywords):
        logger.info("Force recommendation triggered by keywords")
        return {**state, "decision": "recommend_types"}
    
    
    # Force recommendation after 6 turns regardless
    if turn_count >= 6:
        logger.info("Forcing recommendation after 6 turns")
        return {**state, "decision": "recommend_types"}
    
    return {**state, "decision": "continue_question"}

async def generate_recommendation(state: GraphState) -> Dict[str, Any]:
    """Node: Generate structured water heater recommendations"""
    logger.info("Node: generate_recommendation")
    
    try:
        # Add recommendation trigger to conversation
        system_prompt = """
            The user has provided sufficient information about their situation, priorities, and constraints. 
            Now provide comprehensive recommendations for all 6 water heater type systems using this specified format.

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
        
        # Get conversation history
        chat_history = messages_to_langchain_format(state["messages"][:-1])
        current_input = state["messages"][-1].content if state["messages"] else ""
        
        messages = [
            SystemMessage(content=system_prompt.strip()),
            *chat_history,
            HumanMessage(content=current_input.strip())
        ]
        
        # Get streaming response
        full_response = await system_components.llm.ainvoke(messages)
        full_response = full_response.content if hasattr(full_response, 'content') else str(full_response)
        
        is_complete = is_conversation_complete(full_response)
        
        return {
            **state,
            "full_response": full_response,
            "is_complete": is_complete,
            "using_rag": False
        }

    except Exception as e:
        logger.error(f"Error in generate_recommendation: {e}")
        return {
            **state,
            "error": str(e),
            "full_response": "I had trouble generating recommendations.",
            "is_complete": False
        }

async def continue_conversation(state: GraphState) -> Dict[str, Any]:
    """Node: Continue normal conversation flow"""
    logger.info("Node: continue_conversation")
    
    try:
        # Get conversation history
        chat_history = messages_to_langchain_format(state["messages"][:-1])
        current_input = state["messages"][-1].content if state["messages"] else ""
        
        chain_input = {
            "input": current_input,
            "chat_history": chat_history
        }
        
        # Create the chain
        chain = (
            {
                "chat_history": lambda x: x.get("chat_history", []),
                "input": RunnablePassthrough()
            }
            | system_components.chat_prompt
            | system_components.llm
            | system_components.output_parser
        )
        
        # Get full response
        full_response = await chain.ainvoke(chain_input)
        is_complete = is_conversation_complete(full_response)
        
        return {
            "full_response": full_response,
            "is_complete": is_complete,
            "using_rag": False
        }
        
    except Exception as e:
        logger.error(f"Error in continue_conversation: {e}")
        return {
            "error": str(e),
            "full_response": "I apologize, but I encountered an error processing your message.",
            "is_complete": False
        }

def save_response(state: GraphState) -> GraphState:
    """Node: Save assistant response to conversation history"""
    logger.info("Node: save_response")
    
    try:
        # Add the AI response to messages
        ai_message = AIMessage(content=state["full_response"])
        
        # Save to file (optional)
        save_conversation_to_file(state["messages"], state["full_response"])
        
        return {
            **state,
            "messages": state["messages"] + [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Error in save_response: {e}")
        return {
            **state,
            "error": str(e)
        }

#============================================================
#                       UTILITY FUNCTIONS                    
#============================================================

def messages_to_langchain_format(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert messages to LangChain format"""
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage, SystemMessage))]

def has_sufficient_info(messages: List[BaseMessage]) -> bool:
    """Enhanced check for sufficient information"""
    user_messages = [msg.content.lower() for msg in messages if isinstance(msg, HumanMessage)]
    combined_text = ' '.join(user_messages)
    
    # Essential categories with more comprehensive keywords
    essential_info = {
        'situation': ['replace', 'broken', 'new', 'build', 'upgrade', 'old', 'install', 'need', 'want'],
        'infrastructure': ['gas', 'electric', 'line', 'available', 'connection', 'space', 'both', 'electricity', 'natural gas'],
        'household': ['family', 'people', 'person', 'use', 'shower', 'bath', 'dishwasher', 'laundry', 'kids', 'adults'],
        'space': ['space', 'room', 'enough', 'big', 'small', 'basement', 'garage', 'outdoor', 'indoor'],
        'priorities': ['cost', 'save', 'efficient', 'reliable', 'environment', 'important', 'matter', 'budget', 'cheap', 'expensive', 'upfront', 'long term', 'efficiency']
    }
    
    covered_categories = 0
    for category, keywords in essential_info.items():
        if any(keyword in combined_text for keyword in keywords):
            covered_categories += 1
    
    # Need at least 4 out of 5 categories covered
    return covered_categories >= 4

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

#============================================================
#                       GRAPH CONSTRUCTION                   
#============================================================

def create_water_heater_graph():
    """Create and return the LangGraph workflow"""
    
    # Create the graph
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("classify", classify_user_input)
    builder.add_node("rag_query", generate_rag_response)  # This should be async
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
        lambda state: state["classification"]["type"] if state.get("classification") else "answer",
        {
            "question": "rag_query",
            "answer": "add_to_conversation"
        }
    )
    
    # Add regular edges
    builder.add_edge("rag_query", "save_response")
    builder.add_edge("add_to_conversation", "check_recommend")
    
    # Add conditional edges for recommendation check
    builder.add_conditional_edges(
        "check_recommend",
        # lambda state: check_trigger_conditions(state),
        lambda state: state.get("decision", "continue_question"),
        {
            "recommend_types": "recommend",
            "continue_question": "continue"
        }
    )
    
    builder.add_edge("recommend", "save_response")
    builder.add_edge("continue", "save_response")
    builder.add_edge("save_response", END)
    
    # Compile the graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

#============================================================
#                       MAIN INTERFACE                       
#============================================================

class WaterHeaterGraphInterface:
    """Main interface for the Water Heater recommendation system"""
    
    def __init__(self):
        self.graph = create_water_heater_graph()
        self.config = {"configurable": {"thread_id": "default"}}
    
    def start_conversation(self) -> str:
        """Start a new conversation"""
        initial_message = (
            "Hello! I'm here to help you find the perfect water heater type for your home. "
            "To give you the best recommendations, I'd like to understand your situation better.\n\n"
            "What's your main reason for needing a new water heater type? Are you replacing a broken unit, "
            "upgrading for better efficiency, building a new home, or something else?"
        )
        
        # Initialize state
        initial_state = {
            "messages": [AIMessage(content=initial_message)],
            "turn_count": 0,
            "recommendations_given": False,
            "using_rag": False,
            "last_rag_query": None,
            "last_ai_question": None,
            "classification": None,
            "full_response": initial_message,
            "is_complete": False,
            "error": None
        }
        
        return initial_state
    
    async def process_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """Process a user message through the graph with streaming"""
        try:
            # Create human message
            human_msg = HumanMessage(content=user_message)
            session_state = get_conversation_state()
            session_state["messages"].append(human_msg)

            # Initialize streaming response
            full_response = ""
            
            # Process through the graph
            async for event in self.graph.astream(
                session_state, 
                config=self.config
            ):
                # Handle different event types
                for node_name, node_output in event.items():
                    if node_name == "save_response":
                        # Final response handling
                        response = node_output.get("full_response", "")
                        is_complete = node_output.get("is_complete", False)
                        using_rag = node_output.get("using_rag", False)
                        
                        # Update the full response
                        full_response = response
                        
                        # Send final message
                        response_data = {
                            'content': '',
                            'done': True,
                            'full_response': response,
                            'is_complete': is_complete,
                            'using_rag': using_rag
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                        
                    elif node_name in ["continue", "recommend", "rag_query"]:
                        # Stream content as it's generated
                        response = node_output.get("full_response", "")
                        if response:
                            full_response = response
                            rtn_data={
                                'content': response, 
                                'done': False
                            }
                            yield f"data: {json.dumps(rtn_data)}\n\n"
            
            # Update the conversation state with the final response
            if full_response:
                ai_message = AIMessage(content=full_response)
                session_state["messages"].append(ai_message)
                session_state["full_response"] = full_response
                session_state["is_complete"] = is_conversation_complete(full_response)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            rtn_d={
                'error': 'I apologize, but I encountered an error processing your message.'
            }
            yield f"data: {json.dumps(rtn_d)}\n\n"


#============================================================
#                       DEPENDENCIES                         
#============================================================
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
            "error": None
        }
    return app_state.conversation_sessions[session_id]


#============================================================
#                       APP LIFECYCLE                        
#============================================================
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting application...")
    
    try:
        app_state.recommendation_system = WaterHeaterGraphInterface()
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

@app.get("/chat/status")
async def get_chat_status(
    recommendation_system: WaterHeaterGraphInterface = Depends(get_recommendation_system),
    state: GraphState = Depends(get_conversation_state)
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