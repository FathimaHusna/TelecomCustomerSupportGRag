import streamlit as st
import os
import sys
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Telecom Support Chatbot",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.stApp {
    background-color: #f5f7f9;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #e6f7ff;
    border-left: 5px solid #1890ff;
}
.chat-message.assistant {
    background-color: #f6ffed;
    border-left: 5px solid #52c41a;
}
.chat-message .content {
    display: flex;
    margin-top: 0.5rem;
}
.sidebar-content {
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = "unknown"
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"
    
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = os.getenv("DEBUG", "False").lower() == "true"

# Function to check if Ollama is installed and running
API_BASE = os.getenv("API_BASE", "").strip()

# Function to check if Ollama is installed and running
def check_ollama_status():
    try:
        # Check if Ollama service is running
        response = requests.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags", timeout=2)
        if response.status_code == 200:
            # Check if the required model is available
            models = response.json().get("models", [])
            model_name = os.getenv("LOCAL_MODEL_NAME", "llama2")
            
            for model in models:
                if model.get("name") == model_name:
                    return "ready"
            
            return "model_missing"
        else:
            return "not_running"
    except Exception as e:
        print(f"Error checking Ollama status: {e}")
        return "not_running"

# Function to check if FastAPI backend is reachable
def check_api_status():
    base = API_BASE
    if not base:
        return "disabled"
    try:
        resp = requests.get(f"{base.rstrip('/')}/health", timeout=3)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            return "ready" if resp.json().get("initialized") else "initializing"
        return "not_running"
    except Exception:
        return "not_running"

# Function to generate response using Ollama
def generate_ollama_response(prompt):
    try:
        model_name = os.getenv("LOCAL_MODEL_NAME", "llama2")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Call Ollama API
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 800
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response generated")
        else:
            return f"Error generating response: {response.text}"
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

# Function to call the FastAPI backend
def generate_api_response(query):
    try:
        base = API_BASE
        if not base:
            return "API_BASE not configured"
        resp = requests.post(
            f"{base.rstrip('/')}/chat",
            json={"query": query},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "No response")
        return f"Backend error: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"Error calling API: {e}"

# Function to create a prompt for telecom support
def create_telecom_prompt(query):
    system_prompt = """You are a telecom customer support assistant. Your job is to help customers with their telecom-related issues.
    
Provide helpful, accurate, and concise responses to customer queries. Focus on:
1. Identifying the specific issue
2. Providing step-by-step troubleshooting instructions
3. Explaining technical concepts in simple terms
4. Being polite and professional

Common telecom issues include:
- Dropped calls
- Slow internet
- Billing problems
- Signal strength issues
- Data usage concerns

When appropriate, suggest escalation to a human agent for complex issues.
"""
    
    full_prompt = f"{system_prompt}\n\nCustomer: {query}\n\nAssistant:"
    return full_prompt

# Sidebar
with st.sidebar:
    st.title("Telecom Support Chatbot")
    st.image("https://img.icons8.com/color/96/000000/technical-support.png", width=100)
    
    # Backend status indicator (prefer API if configured)
    if API_BASE:
        if st.session_state.api_status == "unknown":
            st.session_state.api_status = check_api_status()
        api_color = {
            "ready": "green",
            "initializing": "orange",
            "not_running": "red",
            "disabled": "gray",
            "unknown": "gray"
        }
        api_message = {
            "ready": f"API backend ready ({API_BASE})",
            "initializing": "API initializing",
            "not_running": "API not reachable",
            "disabled": "API disabled",
            "unknown": "Checking API status..."
        }
        st.markdown(
            f"<div style='display: flex; align-items: center;'>"
            f"<div style='width: 12px; height: 12px; border-radius: 50%; background-color: {api_color[st.session_state.api_status]}; margin-right: 8px;'></div>"
            f"<div>{api_message[st.session_state.api_status]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("Check API"):
            st.session_state.api_status = check_api_status()
            st.rerun()
    else:
        if st.session_state.ollama_status == "unknown":
            status = check_ollama_status()
            st.session_state.ollama_status = status
        status_color = {
            "ready": "green",
            "not_running": "red",
            "model_missing": "orange",
            "unknown": "gray"
        }
        status_message = {
            "ready": "Ollama is ready",
            "not_running": "Ollama not running",
            "model_missing": "Required model missing",
            "unknown": "Checking Ollama status..."
        }
        st.markdown(
            f"<div style='display: flex; align-items: center;'>"
            f"<div style='width: 12px; height: 12px; border-radius: 50%; background-color: {status_color[st.session_state.ollama_status]}; margin-right: 8px;'></div>"
            f"<div>{status_message[st.session_state.ollama_status]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    if st.session_state.ollama_status == "model_missing":
        model_name = os.getenv("LOCAL_MODEL_NAME", "llama2")
        st.warning(f"The required model '{model_name}' is not available.")
        if st.button("Pull Model"):
            with st.spinner(f"Pulling model {model_name}. This may take a few minutes..."):
                import subprocess
                result = subprocess.run(["ollama", "pull", model_name], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success(f"Model {model_name} pulled successfully!")
                    st.session_state.ollama_status = "ready"
                    st.rerun()
                else:
                    st.error(f"Failed to pull model: {result.stderr}")
    
    elif st.session_state.ollama_status == "not_running":
        st.markdown("""#### Ollama Setup
        1. Ensure Ollama service is running
        2. Try restarting Ollama
        """)
        if st.button("Check Again"):
            status = check_ollama_status()
            st.session_state.ollama_status = status
            st.rerun()
    
    st.markdown("---")
    
    # About section
    st.markdown("""
    ## About
    This chatbot uses Ollama (a free, local LLM) to provide telecom customer support without requiring any API keys.
    
    The system is designed to help with common telecom issues like:
    - Dropped calls
    - Slow internet
    - Billing problems
    - Signal strength issues
    - Data usage concerns
    """)
    
    # Initialize button
    if not st.session_state.initialized:
        backend_ready = (
            (API_BASE and st.session_state.api_status in ("ready", "initializing")) or
            (not API_BASE and st.session_state.ollama_status == "ready")
        )
        if backend_ready and st.button("Initialize Chatbot"):
            st.session_state.initialized = True
    
    # Debug mode toggle
    st.markdown("---")
    debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    if debug_mode != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_mode
    
    # Sample queries
    st.markdown("## Sample Queries")
    sample_queries = [
        "My calls keep dropping in the afternoon.",
        "Why is my internet so slow during the evening?",
        "I'm being charged for services I didn't subscribe to.",
        "I have no service in my basement.",
        "My data limit is reached too quickly each month."
    ]
    
    for query in sample_queries:
        if st.button(query, key=f"sample_{query}"):
            st.session_state.messages.append({"role": "user", "content": query})
            if st.session_state.initialized:
                # This will trigger the chat to update in the main area
                st.rerun()
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Please initialize the chatbot first."})
                st.rerun()

# Main area
st.title("Telecom Customer Support")

# Initialize hint
if not st.session_state.initialized:
    st.info("Please initialize the chatbot using the button in the sidebar.")
    if st.button("Initialize Now"):
        st.session_state.initialized = True

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show debug info if enabled and it's an assistant message
        if st.session_state.debug_mode and message["role"] == "assistant" and i > 0:
            # Get the user query that prompted this response
            user_query = st.session_state.messages[i-1]["content"]
            
            with st.expander("Debug Information"):
                st.subheader("Prompt Sent to Ollama")
                st.code(create_telecom_prompt(user_query), language="text")
                
                # Simulate some telecom-specific debug info
                st.subheader("Identified Issue Categories")
                issue_categories = []
                if "call" in user_query.lower() or "drop" in user_query.lower():
                    issue_categories.append("Call Quality")
                if "internet" in user_query.lower() or "slow" in user_query.lower() or "speed" in user_query.lower():
                    issue_categories.append("Internet Performance")
                if "bill" in user_query.lower() or "charge" in user_query.lower() or "payment" in user_query.lower():
                    issue_categories.append("Billing")
                if "signal" in user_query.lower() or "coverage" in user_query.lower() or "reception" in user_query.lower():
                    issue_categories.append("Network Coverage")
                if "data" in user_query.lower() or "usage" in user_query.lower() or "limit" in user_query.lower():
                    issue_categories.append("Data Usage")
                
                if not issue_categories:
                    issue_categories.append("General Inquiry")
                
                for category in issue_categories:
                    st.write(f"- {category}")

# Process last user message if it hasn't been processed yet
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.initialized:
    user_query = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            start_time = time.time()
            if API_BASE:
                response = generate_api_response(user_query)
            else:
                # Create prompt for telecom support
                prompt = create_telecom_prompt(user_query)
                response = generate_ollama_response(prompt)
            end_time = time.time()
            
            st.markdown(response)
            if st.session_state.debug_mode:
                st.caption(f"Response time: {end_time - start_time:.2f} seconds")
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
if st.session_state.initialized:
    prompt = st.chat_input("How can I help you today?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
else:
    # Disabled chat input if not initialized
    st.text_input("How can I help you today?", disabled=True, 
                 placeholder="Please initialize the chatbot first")

# Footer
st.markdown("---")
if API_BASE:
    st.caption("Powered by FastAPI backend + GraphRAG")
else:
    st.caption("Powered by Ollama (Local LLM) - No API key required")

# Add a refresh button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if __name__ == "__main__":
    # Check backend status on startup
    if API_BASE:
        if st.session_state.api_status == "unknown":
            st.session_state.api_status = check_api_status()
    else:
        if st.session_state.ollama_status == "unknown":
            st.session_state.ollama_status = check_ollama_status()
