from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

# Load the conversation data
with open('All_Context/all_three_conversation.json', 'r') as f:
    conversation_data = json.load(f)

# Load the summary table data
with open('All_Context/summary_table.json', 'r') as f:
    summary_table = json.load(f)

# Load the sites info
with open('All_Context/all_sites_info.txt', 'r') as f:
    sites_info = f.read()

# Create a system prompt that incorporates all the context
system_prompt = f"""
You are a water heater expert helping homeowners evaluate different water heating systems. 
Your goal is to have a natural conversation to understand their needs and then provide a ranked comparison of system types.

You have access to:
1. Example conversations: {json.dumps(conversation_data, indent=2)}
2. Water heater data: {json.dumps(summary_table, indent=2)}
3. Technical information: {sites_info}

Guidelines:
1. Start by asking about their main reason for needing a new water heater
2. Ask follow-up questions based on responses
3. Keep conversation natural
4. Ask one question at a time
5. After gathering info, present a ranked comparison table showing:
   - System Type
   - Yearly Cost
   - Affordability Rank
   - Fuel Type
   - Fuel Supply
   - Abundance Rank
   - CO₂ Emissions
   - Environmental Rank
   - Complexity
   - Reliability Rank
6. Highlight which systems best match their stated priorities
7. Finally, Give answer in this rank base system type according to the user's preferences and needs.
   Example:
    ```
        |     System Type      | Affordability(↓ better) | Abundance(↑ better) | Environmental Impact(↑ better) | Reliability(↓ better) |
        | -------------------- | ----------------------- | ------------------- | ------------------------------ | --------------------- |
        | Electric Tank        | 1.57                    | 4.47                | 1.07                           | 4.17                  |
        | Electric Tankless    | 1.00                    | 4.47                | 1.00                           | 2.50                  |
        | Heat Pump            | 2.69                    | 4.47                | 3.92                           | 1.67                  |
        | Natural Gas Tank     | 4.69                    | 4.36                | 1.27                           | 5.00                  |
        | Natural Gas Tankless | 5.00                    | 4.36                | 3.51                           | 3.33                  |
        | Active Solar         | 1.37                    | 5.00                | 5.00                           | 1.00                  |
    ```
Begin by greeting the user and asking about their main reason for needing a new water heater.
"""

client = OpenAI(api_key=openai_api_key)

def get_ai_response(messages):
    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages,
    )
    return response.choices[0].message.content

def main():
    print("Water Heater Comparison System")
    print("---------------------------------")
    print("Hello! I'll help you compare water heater systems based on your needs.")
    print("I'll ask you some questions about your needs and preferences.")
    print("Type 'quit' at any time to end the conversation.\n")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hello! What is your main reason for needing a new water heater?"}
    ]

    print("AI: Hello! What is your main reason for needing a new water heater?")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("AI: Thank you for your time. Feel free to come back if you have more questions about water heaters!")
            break
            
        messages.append({"role": "user", "content": user_input})
        
        ai_response = get_ai_response(messages)
        messages.append({"role": "assistant", "content": ai_response})
        
        print(f"AI: {ai_response}")
        
        # Check if the AI has moved to recommendations
        if "recommend" in ai_response.lower() or "ranking" in ai_response.lower():
            print("\nWould you like to start over? (yes/no)")
            restart = input("You: ").lower()
            if restart == 'yes':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content":"Hello! What is your main reason for needing a new water heater?"}
                ]
                print("AI: Hello! What is your main reason for needing a new water heater?")
            else:
                print("AI: Thank you for your time. Feel free to come back if you have more questions about water heaters!")
                break

if __name__ == "__main__":
    main()