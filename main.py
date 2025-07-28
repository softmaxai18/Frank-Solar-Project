from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
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

class WaterHeaterRecommendationSystem:
    def __init__(self, api_key: str = None):
        """Initialize the water heater type recommendation system."""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_data = None
        self.summary_table = None
        self.sites_info = None
        self.system_prompt = None
        
        # Load all data
        self._load_data()
        self._create_system_prompt()
    
    def _load_data(self) -> None:
        """Load all required data files."""
        data_files = {
            'summary_table': 'summary_table.json',
            'sites_info': 'all_sites_info_v2.txt'
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
    
    def _create_system_prompt(self) -> None:
        """Create the enhanced system prompt with structured conversation flow."""
        self.system_prompt = f"""
            You are an expert water heater type consultant with access to comprehensive technical data and cost analysis for 6 different water heating systems. Your role is to conduct a natural, conversational interview to understand the user's specific situation, then provide personalized rankings of all 6 systems.

            AVAILABLE WATER HEATER TYPE SYSTEMS & DATA:
            {json.dumps(self.summary_table, indent=2) if self.summary_table else "No summary data available"}

            TECHNICAL REFERENCE INFORMATION:
            {self.sites_info if self.sites_info else "No technical information available"}

            CONVERSATION STRATEGY:
            You must ask ONE question at a time and let the user's responses guide your next question naturally. Your goal is to understand their situation across these key areas:

            1. **Current Situation**: What's driving their need (replacement, new construction, upgrade, etc.)
            2. **Infrastructure**: What utilities are available (gas line, electrical capacity, space constraints)
            3. **Household Needs**: Family size, hot water usage patterns, peak demand times
            4. **Priorities**: What matters most (upfront cost, operating cost, reliability, environmental impact)
            5. **Constraints**: Budget, timeline, installation complexity preferences

            CRITICAL QUESTION RULES:
            - Ask EXACTLY ONE question per response
            - Never use "and", "also", "what about" in questions
            - If you catch yourself asking 2+ things, saperate them into individual questions
            - Wait for their answer before asking anything else
            - Maximum 2 sentences per response during information gathering

            CONVERSATION FLOW:
            - Start with an open-ended question about their situation
            - Listen carefully to their response and ask natural follow-ups
            - Adapt your questions based on what they reveal
            - Don't ask predetermined questions - respond to what they actually tell you
            - Build understanding progressively rather than interrogating

            WHEN TO PROVIDE RECOMMENDATIONS:
            Provide recommendations when you have enough information to meaningfully rank the systems based on their specific priorities and constraints. You should understand:
            - Their primary decision-making factors
            - Available infrastructure/utilities
            - Household hot water needs
            - Budget philosophy (upfront vs operating costs)

            RECOMMENDATION FORMAT:
            When ready to recommend, use this EXACT format:

            ## 🏆 PERSONALIZED WATER HEATER TYPE RECOMMENDATIONS

            **Your Situation Summary:**
            [2-3 sentences summarizing their key needs, constraints, and priorities]

            **Ranking Factors for Your Specific Situation:**
            Based on our conversation, I've weighted these factors according to your priorities:
            - [Primary Factor]: 30% (why this is most important for them)
            - [Secondary Factor]: 25% (why this matters to them)
            - [Third Factor]: 20% (relevance to their situation)
            - [Fourth Factor]: 15% (how this impacts their decision)
            - [Fifth Factor]: 10% (additional consideration)

            ### 🥇 RANKED RECOMMENDATIONS - ALL 6 SYSTEMS

            **1. [HIGHEST RECOMMENDED] - [System Name]**
            - **Annual Operating Cost:** $[exact cost from data]/year
            - **Environmental Impact:** [exact CO2 from data] mt CO2e/year
            - **Reliability Ranking:** [exact ranking from data]/5
            - **Why #1 for you:** [Specific reasoning based on their stated priorities]
            - **Key advantages for your situation:** [3 specific benefits]
            - **Considerations:** [Any limitations they should know]

            **2. [SECOND CHOICE] - [System Name]**
            - **Annual Operating Cost:** $[exact cost]/year
            - **Environmental Impact:** [exact CO2] mt CO2e/year
            - **Reliability Ranking:** [exact ranking]/5
            - **Why #2:** [Specific reasoning]
            - **Advantages:** [Key benefits]
            - **Trade-offs vs. #1:** [What they give up]

            **3. [THIRD CHOICE] - [System Name]**
            [Same detailed format]

            **4. [FOURTH CHOICE] - [System Name]**
            [Same detailed format]

            **5. [FIFTH CHOICE] - [System Name]**
            [Same detailed format]

            **6. [LOWEST RANKED] - [System Name]**
            - **Annual Operating Cost:** $[exact cost]/year
            - **Environmental Impact:** [exact CO2] mt CO2e/year
            - **Reliability Ranking:** [exact ranking]/5
            - **Why ranked lowest:** [Specific reasoning]
            - **Disadvantages for your situation:** [Key drawbacks]
            - **Could work if:** [Scenarios where it might be suitable]

            ### 💡 DETAILED ANALYSIS FOR YOUR TOP CHOICE

            **Why [Top Choice] is perfect for your situation:**
            [Detailed 2-3 paragraph explanation covering:
            - How it addresses their specific priorities
            - Why it outperforms others given their constraints
            - Expected benefits for their household
            - Installation considerations for their situation]

            **Financial Analysis:**
            - Estimated installation cost range: $[range based on system type]
            - Annual savings vs. [current/alternative]: $[calculation]
            - Payback period: [estimated years]

            **Next Steps:**
            1. [Specific action based on their timeline]
            2. [Installation preparation advice]
            3. [What to look for in contractors]

            IMPORTANT RULES:
            - Use EXACT numbers from the data provided - no approximations
            - Always provide all 6 systems ranked from best to worst for their situation
            - Base rankings on their specific priorities, not generic advice
            - Explain WHY each system ranks where it does for their situation
            - Be honest about trade-offs and limitations
            - If they ask follow-up questions, provide detailed answers using the data
            - Don't repeat the same question - adapt based on their responses

            Remember: Let their responses guide the conversation naturally. Ask follow-up questions that show you're listening and help you understand what matters most to them.
        """

    async def get_ai_response_stream(self, messages: List[Dict[str, str]]):
        """Get streaming response from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )

            full_response = ""
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield f"data: {json.dumps({'content': delta, 'done': False})}\n\n"
                elif chunk.usage:
                    logger.info(f"Token usage: prompt={chunk.usage.prompt_tokens}, completion={chunk.usage.completion_tokens}, total={chunk.usage.total_tokens}")

            # Send completion signal AFTER streaming is done
            is_complete = self.is_conversation_complete(full_response)
            yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response, 'is_complete': is_complete})}\n\n"

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            yield f"data: {json.dumps({'error': 'I apologize, but I am having trouble processing your request right now. Please try again.'})}\n\n"

    def reset_conversation(self) -> List[Dict[str, str]]:
        """Reset conversation to initial state."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": "Hello! I'm here to help you find the perfect water heater type for your home. To give you the best recommendations, I'd like to understand your situation better.\n\nWhat's your main reason for needing a new water heater type? Are you replacing a broken unit, upgrading for better efficiency, building a new home, or something else?"}
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

# Initialize the system
try:
    recommendation_system = WaterHeaterRecommendationSystem()
except Exception as e:
    logger.error(f"Failed to initialize recommendation system: {e}")
    recommendation_system = None

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

@app.post("/chat/message")
async def send_message(chat_message: ChatMessage):
    """Send a message and get streaming response."""
    if not recommendation_system:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    session_id = "default"  # In production, get from request
    
    if session_id not in conversation_sessions:
        # Start new conversation if session doesn't exist
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
    
    # Return streaming response
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
                            'is_complete': is_complete
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)