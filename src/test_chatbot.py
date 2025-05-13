import os
import sys
from chatbot import TelecomSupportChatbot

def test_chatbot():
    """Test function to verify the telecom support chatbot functionality"""
    print("Testing Telecom Support Chatbot")
    print("==============================")
    
    # Get the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Initialize the chatbot
    print("Initializing chatbot...")
    chatbot = TelecomSupportChatbot(data_dir)
    if not chatbot.initialize():
        print("Failed to initialize chatbot. Please check the data files.")
        return False
    
    print("\nChatbot initialized successfully!")
    
    # Test queries
    test_queries = [
        "My calls keep dropping in the afternoon.",
        "Why is my internet so slow during the evening?",
        "I'm being charged for services I didn't subscribe to.",
        "I have no service in my basement.",
        "My data limit is reached too quickly each month."
    ]
    
    print("\nTesting with sample queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}: '{query}'")
        response = chatbot.generate_response(query)
        print(f"Response: {response}")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    test_chatbot()
