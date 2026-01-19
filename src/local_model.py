import os
import subprocess
import sys
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LocalModelHandler:
    """
    Handler for local LLM models using Ollama
    This provides a completely free alternative that doesn't require any API keys
    """
    
    def __init__(self):
        """Initialize the local model handler"""
        self.model_name = os.getenv("LOCAL_MODEL_NAME", "llama2")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.is_ollama_available = self._check_ollama_available()
        self.timeout = 60  # Timeout in seconds for API calls
        self.force_cpu = os.getenv("OLLAMA_FORCE_CPU", "false").lower() == "true"
        
    def _check_ollama_available(self):
        """Check if Ollama is available on the system"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking Ollama availability: {e}")
            return False
    
    def _check_model_available(self):
        """Check if the specified model is available in Ollama"""
        if not self.is_ollama_available:
            return False
            
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                target = self.model_name
                return any(
                    (m.get("name") == target) or (m.get("name", "").split(":")[0] == target)
                    for m in models
                )
            return False
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def pull_model(self):
        """Pull the specified model if it's not available"""
        try:
            result = subprocess.run(["ollama", "pull", self.model_name], 
                                   capture_output=True, text=True, check=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
    
    def generate_response(self, prompt):
        """
        Generate a response using a local model via Ollama
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            str: Generated response or error message
        """
        # Refresh Ollama availability status
        self.is_ollama_available = self._check_ollama_available()
        
        if not self.is_ollama_available:
            return "Ollama is not available. Please ensure Ollama is running: https://ollama.ai/download"
        
        # Check if model is available, try to pull it if not
        if not self._check_model_available():
            print(f"Model {self.model_name} not available, attempting to pull...")
            if not self.pull_model():
                return f"Model {self.model_name} is not available. Please run 'ollama pull {self.model_name}' first."
        
        def _call_ollama(options_override=None):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 220
                }
            }
            # Global CPU-only hint if set
            if self.force_cpu:
                payload["options"]["num_gpu"] = 0
            if options_override:
                payload["options"].update(options_override)

            start_time = time.time()
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            end_time = time.time()
            print(f"Ollama response time: {end_time - start_time:.2f} seconds")
            return resp

        try:
            # First attempt (default settings)
            response = _call_ollama()
            # If server returns CUDA error, retry with CPU-only hint
            if response.status_code != 200:
                body = response.text or ""
                if "CUDA error" in body or "cuda" in body.lower():
                    print("CUDA error detected from Ollama. Retrying with CPU-only options (num_gpu=0)...")
                    response = _call_ollama({"num_gpu": 0})

            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                # Return helpful guidance if CUDA issues persist
                body = response.text
                if "CUDA" in body or "cuda" in body.lower():
                    return (
                        "Ollama reported a CUDA error. Try CPU fallback: "
                        "restart Ollama with GPU disabled (e.g., set CUDA_VISIBLE_DEVICES='' and run 'ollama serve'), "
                        "or ensure GPU drivers match your CUDA toolchain."
                    )
                return f"Error generating response: {body}"
        except requests.exceptions.Timeout:
            return "Request to Ollama timed out. The model might be too slow or the server is overloaded."
        except Exception as e:
            return f"Error calling local model: {str(e)}"
    
    @staticmethod
    def install_ollama_instructions():
        """
        Return instructions for installing Ollama
        
        Returns:
            str: Installation instructions
        """
        return """
To use local models without any API key, you need to install Ollama:

1. Visit https://ollama.ai/download and follow the installation instructions for your OS.
2. After installation, pull a model: `ollama pull llama2` (or another model of your choice)
3. Start the Ollama service if it's not running automatically
4. Restart the application

Ollama provides completely free local inference without requiring any API keys.
"""
