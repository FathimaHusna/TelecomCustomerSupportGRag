import os
import sys
from chatbot import TelecomSupportChatbot

def main():
    """Main function to run the telecom support chatbot from the command line"""
    print("Telecom Support Chatbot")
    print("======================")
    
    # Get the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Initialize the chatbot
    print("Initializing chatbot...")
    chatbot = TelecomSupportChatbot(data_dir)
    if not chatbot.initialize():
        print("Failed to initialize chatbot. Please check the data files.")
        return
    
    print("\nChatbot initialized successfully!")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    # Main conversation loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Thank you for using the Telecom Support Chatbot. Goodbye!")
            break
        
        # Generate response
        response = chatbot.generate_response(user_input)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    main()
