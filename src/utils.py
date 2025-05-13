import os
import json
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chatbot.log'))
    ]
)

logger = logging.getLogger('telecom_chatbot')

class ChatbotUtils:
    """Utility functions for the telecom support chatbot"""
    
    @staticmethod
    def create_directories():
        """Create necessary directories for the chatbot"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        dirs = [
            os.path.join(base_dir, "data"),
            os.path.join(base_dir, "docs"),
            os.path.join(base_dir, "exports"),
            os.path.join(base_dir, "evaluations"),
            os.path.join(base_dir, "logs")
        ]
        
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    @staticmethod
    def log_query(query, response, query_time=None):
        """Log a query and response to the query log file"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, "query_log.jsonl")
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "query_time": query_time
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    @staticmethod
    def visualize_subgraph(graph, central_node, depth=2, output_path=None):
        """
        Visualize a subgraph centered around a specific node
        
        Args:
            graph (nx.Graph): The full knowledge graph
            central_node (str): The central node to focus on
            depth (int): How many hops away from the central node to include
            output_path (str): Path to save the visualization
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        # Extract subgraph
        nodes = {central_node}
        current_nodes = {central_node}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                if node in graph:
                    next_nodes.update(graph.neighbors(node))
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        subgraph = graph.subgraph(nodes)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph, seed=42)
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', 'unknown')
            if node_type == 'issue':
                node_colors.append('red')
            elif node_type == 'cause':
                node_colors.append('green')
            elif node_type == 'resolution':
                node_colors.append('blue')
            elif node_type == 'device':
                node_colors.append('orange')
            elif node_type == 'network':
                node_colors.append('purple')
            elif node_type == 'manual':
                node_colors.append('brown')
            else:
                node_colors.append('gray')
        
        # Highlight the central node
        node_sizes = [300 if node == central_node else 100 for node in subgraph.nodes()]
        
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, alpha=0.8, node_size=node_sizes)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, arrows=True)
        
        # Add labels
        labels = {}
        for node in subgraph.nodes():
            # Truncate long labels
            labels[node] = str(node)[:20] + '...' if len(str(node)) > 20 else str(node)
        
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
        
        plt.title(f"Knowledge Graph around '{central_node}'")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Subgraph visualization saved to {output_path}")
        
        return plt.gcf()
    
    @staticmethod
    def timer(func):
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
            return result
        return wrapper
