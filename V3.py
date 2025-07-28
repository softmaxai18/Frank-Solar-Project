from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class WaterHeaterRecommendationSystem:
    def __init__(self, api_key: str = None):
        """Initialize the water heater recommendation system."""
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
                    raise
    
    def _create_system_prompt(self) -> None:
        """Create the enhanced system prompt with structured conversation flow."""
        self.system_prompt = f"""
            You are an expert water heater consultant with access to comprehensive technical data and cost analysis for 6 different water heating systems. Your role is to conduct a natural, conversational interview to understand the user's specific situation, then provide personalized rankings of all 6 systems.

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

            CONVERSATION STARTER:
            "I'm here to help you find the perfect water heater type for your specific situation. Every household is different, so I'd love to understand what's going on with your current setup or what's driving your need for a new water heating system type."

            Remember: Let their responses guide the conversation naturally. Ask follow-up questions that show you're listening and help you understand what matters most to them.
        """

    def get_ai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from OpenAI API with error handling."""
        try:
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}  # Optional: get usage in stream
            )

            full_response = ""
            token_usage = None

            print("\n💬 Consultant: ", end="", flush=True)  # start the response line

            for chunk in response:
                # Handle streamed content
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        print(delta, end="", flush=True)
                # Handle final usage info
                elif chunk.usage:
                    token_usage = chunk.usage

            print()  # ensure newline at the end of streaming

            if token_usage:
                logger.info(
                    f"Token usage: prompt={token_usage.prompt_tokens}, "
                    f"completion={token_usage.completion_tokens}, "
                    f"total={token_usage.total_tokens}"
                )

            return full_response

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
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
        
        # Check for essential information categories
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
        
        # Need at least 3 categories covered AND priorities mentioned
        return covered_categories >= 3 and priority_mentioned
    
    def run(self) -> None:
        """Run the main conversation loop with enhanced logic."""
        print("=" * 60)
        print("🔥 ADVANCED WATER HEATER TYPE RECOMMENDATION SYSTEM 🔥")
        print("=" * 60)
        print("I'll help you find the perfect water heater type through a personalized consultation.")
        print("I'll ask targeted questions to understand your specific needs and situation.")
        print("Type 'quit' to exit, 'restart' to start over, or 'recommend' to force recommendations.\n")
        
        messages = self.reset_conversation()
        print("💬 Consultant: Hello! I'm here to help you find the perfect water heater type for your home. To give you the best recommendations, I'd like to understand your situation better.\n\nWhat's your main reason for needing a new water heater type? Are you replacing a broken unit, upgrading for better efficiency, building a new home, or something else?")

        conversation_turn = 0
        
        while True:
            try:
                user_input = input("\n🏠 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n💬 Consultant: Thank you for using our Water Heater type Recommendation System! Feel free to return anytime for expert advice. Have a great day! 👋")
                    break
                
                if user_input.lower() in ['restart', 'start over', 'reset']:
                    messages = self.reset_conversation()
                    conversation_turn = 0
                    print("\n💬 Consultant: Hello! I'm here to help you find the perfect water heater type for your home. To give you the best recommendations, I'd like to understand your situation better.\n\nWhat's your main reason for needing a new water heater type? Are you replacing a broken unit, upgrading for better efficiency, building a new home, or something else?")
                    continue
                
                if user_input.lower() in ['recommend', 'give recommendations', 'show options']:
                    # Force recommendations with current information
                    force_recommendation_msg = "Based on the information provided so far, please provide your detailed recommendations using the specified format."
                    messages.append({"role": "user", "content": force_recommendation_msg})
                    ai_response = self.get_ai_response(messages)
                    messages.append({"role": "assistant", "content": ai_response})
                    print(f"\n💬 Consultant: {ai_response}")
                    continue
                
                if not user_input:
                    print("Please share your thoughts or type 'quit' to exit.")
                    continue
                
                messages.append({"role": "user", "content": user_input})
                conversation_turn += 1
                
                # Check if we should move to recommendations
                # In the main run loop, replace the recommendation trigger section:
                if conversation_turn >= 4 and self.has_sufficient_info(messages):
                    # Add a system message to trigger recommendations
                    recommendation_trigger = """
                    The user has provided sufficient information about their situation, priorities, and constraints. 
                    Now provide comprehensive recommendations for all 6 water heater type systems using the specified format.
                    Rank them based on the user's specific situation and priorities discussed.
                    """
                    messages.append({"role": "system", "content": recommendation_trigger})
                
                ai_response = self.get_ai_response(messages)
                messages.append({"role": "assistant", "content": ai_response})
                
                # print(f"\n💬 Consultant: {ai_response}")
                
                # Check if conversation is complete
                if self.is_conversation_complete(ai_response):
                    print("\n" + "="*60)
                    print("🎯 CONSULTATION COMPLETE!")
                    print("Would you like to:")
                    print("1. Start a new consultation (type 'restart')")
                    print("2. Ask follow-up questions about these recommendations")
                    print("3. Exit (type 'quit')")
                    print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n💬 Consultant: Thank you for using our recommendation system! Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                print(f"An error occurred: {e}")
                print("Type 'restart' to start over or 'quit' to exit.")

def main():
    """Main entry point."""
    try:
        system = WaterHeaterRecommendationSystem()
        system.run()
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        print(f"Error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()