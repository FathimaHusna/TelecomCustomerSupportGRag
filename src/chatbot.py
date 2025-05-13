import os
import sys
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path

from local_model import LocalModelHandler
from graph_rag import GraphRAG

# Load environment variables
load_dotenv()

class TelecomSupportChatbot:
    """
    A chatbot for telecom customer support that uses Graph RAG to provide context-aware responses
    """
    
    def __init__(self, data_dir):
        """
        Initialize the chatbot with the data directory
        
        Args:
            data_dir (str): Path to the directory containing the data files
        """
        self.data_dir = data_dir
        self.graph_rag = None
        self.local_model = LocalModelHandler()
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.enable_visualization = os.getenv("ENABLE_VISUALIZATION", "False").lower() == "true"
        
    def initialize(self):
        """
        Initialize the chatbot by loading the knowledge graph and setting up the LLM
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Check if Ollama is available
            if not self.local_model.is_ollama_available:
                print("Ollama is not available. Please ensure Ollama is running.")
                return False
                
            # Initialize GraphRAG
            self.graph_rag = GraphRAG(self.data_dir)
            success = self.graph_rag.initialize()
            
            if not success:
                print("Failed to initialize GraphRAG")
                return False
            
            # Visualize the graph if enabled
            if self.enable_visualization:
                self._visualize_graph()
                
            return True
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            return False
    
    def _visualize_graph(self):
        """
        Visualize the knowledge graph and save it to a file
        """
        try:
            # Create the docs directory if it doesn't exist
            docs_dir = os.path.join(os.path.dirname(self.data_dir), "docs")
            os.makedirs(docs_dir, exist_ok=True)
            
            # Get the graph from GraphRAG
            G = self.graph_rag.graph
            
            # Create a figure and axis
            plt.figure(figsize=(12, 8))
            
            # Create a layout for the nodes
            pos = nx.spring_layout(G, seed=42)
            
            # Draw the nodes
            node_colors = []
            for node in G.nodes():
                if G.nodes[node].get('type') == 'issue':
                    node_colors.append('lightblue')
                elif G.nodes[node].get('type') == 'cause':
                    node_colors.append('lightcoral')
                elif G.nodes[node].get('type') == 'resolution':
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('gray')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
            
            # Draw the edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            
            # Draw the labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            
            # Save the figure
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(docs_dir, "knowledge_graph.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Knowledge graph visualization saved to docs/knowledge_graph.png")
        except Exception as e:
            print(f"Error visualizing graph: {e}")
    
    def generate_response(self, query):
        """
        Generate a response to the user's query
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The chatbot's response
        """
        try:
            # Query the graph to get relevant context
            graph_results = self.graph_rag.query_graph(query)
            
            # Construct the prompt with the graph context
            prompt = self._construct_prompt(query, graph_results)
            
            # Generate the response using the local model
            response = self.local_model.generate_response(prompt)
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def _construct_prompt(self, query, graph_results):
        """
        Construct a prompt for the LLM using the graph results
        
        Args:
            query (str): The user's query
            graph_results (dict): Results from querying the graph
            
        Returns:
            str: The constructed prompt
        """
        prompt = f"""You are a telecom customer support assistant. Use the following information to provide a helpful, accurate, and concise response to the customer's query. 

Customer Query: {query}

"""
        context = """
        
# Relevant Information from Knowledge Graph:
"""
        
        # Add information about identified issues
        if graph_results['entities']['issue_types']:
            context += "Identified issues: " + ", ".join(graph_results['entities']['issue_types']) + "\n\n"
        
        # Add potential causes
        if graph_results['causes']:
            context += "Potential causes:\n"
            for cause in set(graph_results['causes']):
                context += f"- {cause}\n"
            context += "\n"
        
        # Add recommended resolutions
        if graph_results['resolutions']:
            context += "Recommended resolutions:\n"
            for resolution in set(graph_results['resolutions']):
                context += f"- {resolution}\n"
            context += "\n"
        
        # Add relevant paths (issue -> cause -> resolution)
        if graph_results['paths']:
            context += "Relevant troubleshooting paths:\n"
            for path in graph_results['paths']:
                if len(path) >= 3:
                    context += f"- Issue: {path[0]} → Cause: {path[1]} → Resolution: {path[2]}\n"
                else:
                    context += f"- Path: {' → '.join(path)}\n"
            context += "\n"
        
        # Add information from technical manuals if available
        if 'relevant_manuals' in graph_results and graph_results['relevant_manuals']:
            context += "Relevant technical information:\n"
            for manual in graph_results['relevant_manuals']:
                context += f"- {manual['title']}: {manual['content'][:200]}...\n"
            context += "\n"
        
        # Add similar support tickets if available
        if 'similar_docs' in graph_results and graph_results['similar_docs']:
            context += "Similar support tickets:\n"
            for i, (doc, metadata, score) in enumerate(graph_results['similar_docs'][:3]):
                if metadata.get('source') == 'support_ticket':
                    context += f"- Ticket {metadata.get('id', i)}: {metadata.get('issue_type', 'Unknown')} - {metadata.get('resolution', 'No resolution recorded')}\n"
            context += "\n"
        
        # Add instructions for the model
        context += """
Based on the above information, please provide:
1. A clear explanation of the likely cause of the customer's issue
2. Step-by-step troubleshooting instructions
3. When to escalate to a human agent if needed
4. A polite and professional tone throughout

Remember to focus only on the customer's specific issue and avoid unnecessary information.
"""
        
        # Combine prompt with context
        full_prompt = prompt + context
        return full_prompt
