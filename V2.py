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
            'conversation_data': 'all_three_conversation.json',
            'summary_table': 'summary_table.json',
            'sites_info': 'all_sites_info.txt'
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
        """Create the system prompt with all context."""
        self.system_prompt = f"""
        You are a water heater expert helping homeowners evaluate different water heating systems. 
        Your goal is to have a natural conversation to understand their needs and then provide a ranked comparison of system types.

        You have access to:
        1. Example conversations: {json.dumps(self.conversation_data, indent=2) if self.conversation_data else "No conversation examples available"}
        2. Water heater data: {json.dumps(self.summary_table, indent=2) if self.summary_table else "No summary data available"}
        3. Technical information: {self.sites_info if self.sites_info else "No technical information available"}

        Guidelines:
        1. Start by asking about their main reason for needing a new water heater
        2. Ask follow-up questions based on responses (current system, household size, location, priorities)
        3. Keep conversation natural and engaging
        4. Ask one question at a time to avoid overwhelming the user, don't ask multiple questions in one message
        5. Gather key information: current system type/age, household size, location, fuel availability, budget, priorities
        6. After gathering sufficient info, present a ranked comparison table showing:
        - System Type
        - Yearly Cost
        - Affordability Rank (lower is better)
        - Fuel Type
        - Fuel Supply (years)
        - Abundance Rank (higher is better)
        - CO₂ Emissions (mt/yr)
        - Environmental Rank (higher is better)
        - Complexity (1-6 scale)
        - Reliability Rank (lower is better)
        7. Highlight which systems best match their stated priorities
        8. Provide specific recommendations based on their unique situation
        9. Don't ask too many questions, just ask the questions that are necessary to understand their needs and preferences.
        Begin by greeting the user and asking about their main reason for needing a new water heater.
        """
    
    def get_ai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from OpenAI API with error handling."""
        try:
            response = self.client.chat.completions.create(
                model="o4-mini",  # Fixed model name
                messages=messages,
                # max_completion_tokens=1200,  # Increased for better responses
            )

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            logger.info(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    def reset_conversation(self) -> List[Dict[str, str]]:
        """Reset conversation to initial state."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": "Hello! I'm here to help you find the perfect water heater for your home. What is your main reason for needing a new water heater?"}
        ]   
    
    def is_conversation_complete(self, response: str) -> bool:
        """Check if the conversation has reached the recommendation stage."""
        completion_indicators = [
            "recommend", "ranking", "comparison table", "based on your needs",
            "here's my analysis", "summary of options", "best options for you"
        ]
        return any(indicator in response.lower() for indicator in completion_indicators)
    
    def run(self) -> None:
        """Run the main conversation loop."""
        print("=" * 50)
        print("🔥 Water Heater Recommendation System 🔥")
        print("=" * 50)
        print("I'll help you compare water heater systems based on your specific needs.")
        print("I'll ask you some questions about your situation and preferences.")
        print("Type 'quit' at any time to exit, or 'restart' to start over.\n")
        
        messages = self.reset_conversation()
        print("AI: Hello! I'm here to help you find the perfect water heater for your home. What is your main reason for needing a new water heater?")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nAI: Thank you for using the Water Heater Recommendation System! Feel free to come back anytime if you have more questions about water heaters. Have a great day! 👋")
                    break
                
                if user_input.lower() in ['restart', 'start over', 'reset']:
                    messages = self.reset_conversation()
                    print("\nAI: Hello! I'm here to help you find the perfect water heater for your home. What is your main reason for needing a new water heater?")
                    continue
                
                if not user_input:
                    print("Please enter your response or type 'quit' to exit.")
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                ai_response = self.get_ai_response(messages)
                messages.append({"role": "assistant", "content": ai_response})
                
                print(f"\nAI: {ai_response}")
                
                # Check if conversation is complete
                if self.is_conversation_complete(ai_response):
                    print("\n" + "="*50)
                    print("Would you like to:")
                    print("1. Start a new consultation (type 'restart')")
                    print("2. Ask follow-up questions (continue typing)")
                    print("3. Exit (type 'quit')")
                    print("="*50)
                
            except KeyboardInterrupt:
                print("\n\nAI: Goodbye! Thanks for using the Water Heater Recommendation System!")
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

